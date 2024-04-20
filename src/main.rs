#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    ops::{log_softmax, softmax},
    Conv2d, Conv2dConfig, Module, VarBuilder,
};
use candle_transformers::models::t5::{self, T5EncoderModel};
use serde_json;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "tracing")]
use tracing_chrome::ChromeLayerBuilder;
#[cfg(feature = "tracing")]
use tracing_subscriber::prelude::*;
use pico_args::Arguments;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

const DTYPE: DType = DType::F16;
const BIT_FACTOR: f32 = 8.0;
const SCORE_BIAS: f32 = 0.0;
const PROFILE_AA_SIZE: i32 = 20;

struct Args {
    cpu: bool,
    disable_cache: bool,
    prompt: Option<String>,
    generate_profile: bool,
    output: Option<String>,
}

fn predict(
    prompt: String,
    model: &mut T5EncoderModel,
    cnn: &CNN,
    _profile_cnn: &CNN,
    hashmap: &HashMap<String, usize>,
    device: &Device,
    profile: bool,
    buffer_file: &mut File,
    index_file: &mut File,
    buffer_file_seqs: &mut File,
    index_file_seqs: &mut File,
    embeds_file: &mut File,
    index: i32,
    curr_len: i32,
    seq_len: i32,
) -> Result<(Vec<String>, i32)> {
    let copy = &prompt;
    println!("{:?}", copy);
    // Replace each character in the string
    let replaced_values: Vec<Option<&usize>> =
        copy.chars().map(|c| hashmap.get(&c.to_string())).collect();

    let unknown_value: usize = 2; // Default value for None

    let ts: Vec<&usize> = replaced_values
        .iter()
        .map(|option| option.unwrap_or(&unknown_value))
        .collect();
    let mut tokens: Vec<&usize> = vec![hashmap.get("<AA2fold>").unwrap()];
    tokens.extend(ts.iter().clone());
    tokens.push(hashmap.get("</s>").unwrap());
    let tokens: Vec<i64> = tokens.iter().map(|&num| *num as i64).collect();

    let input_token_ids = Tensor::new(&tokens[..], device)?
        .unsqueeze(0)?
        .to_dtype(DType::U8)?;

    let ys = model.forward(&input_token_ids)?;

    let ys = ys.i((.., 1..ys.dims3()?.1 - 1))?;
    let ys = ys.pad_with_zeros(1, 0, 1)?;
    let _ = writeln!(embeds_file, "{}", ys);
    // println!("9");
    let output = &cnn.forward(&ys)?;
    println!("{:?}", output.shape());
    let mut structures: Vec<String> = Vec::new();

    let vals = output.argmax_keepdim(1)?;
    let vals = vals.flatten(0, 2)?;
    let vals = vals.to_vec1::<u32>()?;
    let structure: Vec<char> = vals.iter().map(|&n| number_to_char(n)).collect();

    if profile {
        println!("generating profile");
        //let output = &cnn.forward(&ys)?;
        let probs: Tensor = softmax(output, 1)?;
        let lines: Vec<Vec<f32>> = probs.to_dtype(DType::F32)?.to_vec3()?[0].to_vec();

        let mut profile: Vec<f32> = Vec::new();
        for i in 0..lines[0].len() {
            for j in 0..20 {
                profile.push(lines[j][i]);
                //print!("{:?} ", lines[j][i]);
            }
            //print!("\n");
        }

        let consensus = structure.clone();
        let mut pssm = compute_log_pssm(profile, structure.len())?;
        for pos in 0..consensus.len() {
            if consensus[pos] == 'X' {
                for aa in 0..PROFILE_AA_SIZE {
                    pssm[pos * PROFILE_AA_SIZE as usize + aa as usize] = -1;
                }
            }
        }
        let result = to_buffer(consensus, structure.len(), pssm)?;

        //buffer_file.write_all(result_as_u8)?;

        buffer_file.write(&result)?;

        //write!(buffer_file, "{}",  &result)?;

        let index_content = format!("{}\t{}\t{}\n", index, curr_len, result.len());
        index_file.write(index_content.as_bytes())?;

        write!(buffer_file_seqs, "{}\n\0", &prompt)?;

        let index_content = format!("{}\t{}\t{}\n", index, seq_len, &prompt.len() + 2);
        index_file_seqs.write(index_content.as_bytes())?;

        let structure: String = structure.into_iter().collect();
        structures.push(structure.clone());
        Ok((structures, result.len() as i32))
    } else {
        let structure: String = structure.into_iter().collect();
        structures.push(structure.clone());

        Ok((structures, 0))
    }
}

fn process_fasta(
    input_path: &Path,
    output_path: &Path,
    model: &mut T5EncoderModel,
    cnn: &CNN,
    profile_cnn: &CNN,
    hashmap: &HashMap<String, usize>,
    device: &Device,
    profile: bool,
) -> io::Result<()> {
    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;
    let mut time_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./times.fasta")?;
    let mut buffer_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_ss")?;
    let mut index_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_ss.index")?;
    let mut dbtype_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_ss.dbtype")?;
    let mut buffer_file_seqs = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile")?;
    let mut index_file_seqs = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile.index")?;
    let mut dbtype_file_seqs = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile.dbtype")?;
    let mut buffer_file_h = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_h")?;
    let mut index_file_h = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_h.index")?;
    let mut dbtype_file_h = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile_h.dbtype")?;
    let mut lookup_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./profile.lookup")?;
    let mut embeds_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("./embeds.txt")?;
    let u8_vec_ss: Vec<u8> = vec![2, 0, 0, 0];
    let u8_vec_aa: Vec<u8> = vec![0, 0, 0, 0];
    let u8_vec_h: Vec<u8> = vec![12, 0, 0, 0];
    dbtype_file.write(&u8_vec_ss)?;
    dbtype_file_seqs.write(&u8_vec_aa)?;
    dbtype_file_h.write(&u8_vec_h)?;
    let mut s = String::new();
    let mut index = 0;
    let mut curr_len = 0;
    let mut seq_len = 0;
    let mut header_len = 0;

    for line in reader.lines() {
        let line = line?;

        if line.starts_with('>') {
            let l = line.clone();

            write!(buffer_file_h, "{}\n\0", &l[1..])?;
            let index_content = format!("{}\t{}\t{}\n", index, header_len, l.len() + 1);
            index_file_h.write(index_content.as_bytes())?;
            header_len += (l.len() + 1) as i32;
            let index_content = format!("{}\t{}\t{}\n", index, &l[1..], 0);
            lookup_file.write(index_content.as_bytes())?;
            if !s.is_empty() {
                let start_time = std::time::Instant::now();
                let prompt = s.clone();
                let prediction = predict(
                    prompt,
                    model,
                    cnn,
                    profile_cnn,
                    hashmap,
                    device,
                    profile,
                    &mut buffer_file,
                    &mut index_file,
                    &mut buffer_file_seqs,
                    &mut index_file_seqs,
                    &mut embeds_file,
                    index,
                    curr_len,
                    seq_len,
                );
                let prediction = prediction.unwrap();
                let res: Vec<String> = prediction.0;

                curr_len += prediction.1;
                seq_len += (s.len() + 2) as i32;
                println!("Took {:?}", start_time.elapsed());
                for mut value in res {
                    value.pop();
                    let _ = writeln!(output_file, "{}", value);
                    let _ = writeln!(
                        time_file,
                        "{} : {}",
                        s.len(),
                        start_time.elapsed().as_secs() as f64
                            + start_time.elapsed().subsec_nanos() as f64 * 1e-9
                    );
                }
                s.clear();
            }
            writeln!(output_file, "{}", line)?;
        } else {
            index += 1;
            s.push_str(&line);
        }
    }

    // Write the last sequence if not empty
    if !s.is_empty() {
        let prompt = s.clone();
        let prediction = predict(
            prompt,
            model,
            cnn,
            profile_cnn,
            hashmap,
            device,
            profile,
            &mut buffer_file,
            &mut index_file,
            &mut buffer_file_seqs,
            &mut index_file_seqs,
            &mut embeds_file,
            index,
            curr_len,
            seq_len,
        );
        let res: Vec<String> = prediction.unwrap().0;
        for value in &res {
            let _ = writeln!(output_file, "{}", value);
        }
    }

    Ok(())
}

fn compute_log_pssm(profile: Vec<f32>, query_length: usize) -> Result<Vec<i8>> {
    let pback: Vec<f32> = vec![
        0.0489372, 0.0306991, 0.101049, 0.0329671, 0.0276149, 0.0416262, 0.0452521, 0.030876,
        0.0297251, 0.0607036, 0.0150238, 0.0215826, 0.0783843, 0.0512926, 0.0264886, 0.0610702,
        0.0201311, 0.215998, 0.0310265, 0.0295417, 0.00001,
    ];
    //let pback: Vec<f32> = vec![0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.00001];
    let mut pssm: Vec<i8> = vec![0; query_length * PROFILE_AA_SIZE as usize];
    println!("{query_length}");
    println!("{}", profile.len());
    for pos in 0..query_length {
        for aa in 0..PROFILE_AA_SIZE {
            let aa_prob: f32 = profile[pos * PROFILE_AA_SIZE as usize + aa as usize];
            let idx: usize = pos * PROFILE_AA_SIZE as usize + aa as usize;
            let log_prob: f32 = (aa_prob / pback[aa as usize]).log2();
            let mut pssmval: f32 = log_prob * BIT_FACTOR + BIT_FACTOR * SCORE_BIAS;
            if pssmval < 0.0 {
                pssmval -= 0.5;
            } else {
                pssmval += 0.5;
            }
            if pssmval > 127.0 {
                pssmval = 127.0;
            }
            if pssmval < -128.0 {
                pssmval = -128.0;
            }
            pssm[idx] = pssmval as i8;
        }
    }
    Ok(pssm)
}

fn to_buffer(consensus: Vec<char>, center_sequence_len: usize, pssm: Vec<i8>) -> Result<Vec<u8>> {
    let pssm_u8 = unsafe { std::slice::from_raw_parts(pssm.as_ptr() as *const u8, pssm.len()) };
    // for prob in pssm_u8 {
    //     print!("{prob}\n");
    // }
    let mut result: Vec<u8> = Vec::new();
    for pos in 0..center_sequence_len - 1 {
        for aa in 0..PROFILE_AA_SIZE {
            result.push(pssm_u8[pos * PROFILE_AA_SIZE as usize + aa as usize]);
        }
        result.push(char_to_number(consensus[pos]));
        result.push(char_to_number(consensus[pos]));
        result.push(0 as u8);
        result.push(0 as u8);
        result.push(0 as u8);
    }
    result.push(0 as u8);
    Ok(result)
}

fn number_to_char(n: u32) -> char {
    match n {
        0 => 'A',
        1 => 'C',
        2 => 'D',
        3 => 'E',
        4 => 'F',
        5 => 'G',
        6 => 'H',
        7 => 'I',
        8 => 'K',
        9 => 'L',
        10 => 'M',
        11 => 'N',
        12 => 'P',
        13 => 'Q',
        14 => 'R',
        15 => 'S',
        16 => 'T',
        17 => 'V',
        18 => 'W',
        19 => 'Y',
        _ => 'X', // Default case for numbers not in the list
    }
}

fn char_to_number(n: char) -> u8 {
    match n {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        _ => 20, // Default case for numbers not in the list
    }
}

pub fn conv2d_non_square(
    in_channels: usize,
    out_channels: usize,
    kernel_size1: usize,
    kernel_size2: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size1,
            kernel_size2,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub struct CNN {
    conv1: Conv2d,
    // act: Activation,
    // dropout: Dropout,
    conv2: Conv2d,
    profile: bool,
}

impl CNN {
    pub fn load(vb: VarBuilder, config: Conv2dConfig, profile: bool) -> Result<Self> {
        let conv1 = conv2d_non_square(1024, 32, 7, 1, config, vb.pp("classifier.0"))?;
        // let act = Activation::Relu;
        // let dropout = Dropout::new(0.0);
        let conv2 = conv2d_non_square(32, 20, 7, 1, config, vb.pp("classifier.3"))?;
        // Ok(Self { conv1, act, dropout, conv2 })
        Ok(Self { conv1, conv2, profile })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //println!("input shape: {:?}", xs.shape());
        let xs: Tensor = xs.permute((0, 2, 1))?.unsqueeze(D::Minus1)?;
        //println!("input after permutation: ");
        //println!("{xs}");
        //println!("{:?}", xs.shape());
        // println!("{:?}", xs.shape());
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        // println!("{:?}", xs.shape());
        let xs: Tensor = self.conv1.forward(&xs)?;
        // println!("{:?}", xs.shape());
        // println!("{xs}");
        //println!("{:?}", xs.shape());
        let xs: Tensor = xs.relu()?;
        let xs: Tensor = xs.clone();
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        let xs = self.conv2.forward(&xs)?.squeeze(D::Minus1)?;
        if self.profile {
            let xs: Tensor = xs.permute((0, 2, 1))?;
            let xs = log_softmax(&xs, D::Minus1)?;
            Ok(xs)
        } else {
            Ok(xs)
        }
    }
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
    cnn_filename: Vec<PathBuf>,
    profile_filename: Vec<PathBuf>,
    tokens_map: HashMap<String, usize>,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<Self> {
        let device = device(args.cpu)?;

        let config_filename = PathBuf::from("model/config.json");
        let weights_filename = vec![PathBuf::from("model/model.safetensors")];
        let path = PathBuf::from("model/cnn.safetensors");
        let profile_path = PathBuf::from("model/new_cnn.safetensors");
        //let cnn_filename = path;
        let cnn_filename = vec![path];
        let profile_filename = vec![profile_path];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;

        let tokens_filename = Path::new("model/tokens.json");
        let tokens_config = std::fs::read_to_string(tokens_filename)?;
        let tokens_map: HashMap<String, usize> = serde_json::from_str(&tokens_config)?;

        Ok(Self {
            device,
            config,
            weights_filename,
            cnn_filename,
            profile_filename,
            tokens_map
        })
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

    pub fn build_cnn(&self) -> Result<CNN> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.cnn_filename, DTYPE, &self.device)?
        };
        //println!("varbuilder initialized!");
        let config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(CNN::load(vb, config, false)?)
    }

    pub fn build_profile_cnn(&self) -> Result<CNN> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.profile_filename, DTYPE, &self.device)?
        };
        //println!("varbuilder initialized!");
        let config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(CNN::load(vb, config, true)?)
    }

    // pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
    //     let vb = unsafe {
    //         VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
    //     };
    //     Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    // }
}

fn main() -> Result<()> {
    let mut args = Arguments::from_env();

    // Convert the argument parsing to manually handle each option
    let cpu = args.contains("--cpu");
    let disable_cache = args.opt_value_from_str("--disable-cache").unwrap_or(Some(false)).unwrap_or(false);
    let prompt = args.opt_value_from_str("--prompt").unwrap_or(None);
    let generate_profile = args.opt_value_from_str("--generate-profile").unwrap_or(Some(false)).unwrap_or(false);
    let output = args.opt_value_from_str("--output").unwrap_or(None);

    // Construct Args
    let args = Args {
        cpu,
        disable_cache,
        prompt,
        generate_profile,
        output,
    };

    #[cfg(feature = "tracing")]
    let _guard = {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    };

    let builder = T5ModelBuilder::load(&args)?;
    let device = &builder.device;

    let mut model = builder.build_encoder()?;
    let cnn: &CNN = &builder.build_cnn()?;
    let profile_cnn = &builder.build_profile_cnn()?;
    match args.prompt {
        Some(prompt) => {
            let output = args.output.unwrap();
            let start = std::time::Instant::now();
            let _ = process_fasta(
                Path::new(&prompt),
                Path::new(&output),
                &mut model,
                cnn,
                profile_cnn,
                &builder.tokens_map,
                device,
                args.generate_profile,
            );
            println!("Took {:?}", start.elapsed());
            //println!("finished");
            //let res = predict(prompt, &builder, &hashmap, device)?;
            //println!{"{:?}", res};
        }
        None => {}
    }
    Ok(())
}
