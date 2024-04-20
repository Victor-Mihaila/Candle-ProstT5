#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "cuda")]
extern crate cudarc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::{Module, VarBuilder, Conv2d, Conv2dConfig, ops::{softmax, log_softmax}};
use candle_transformers::models::t5::{self, T5EncoderModel};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write, Read};
use std::path::Path;
use std::path::PathBuf;
// use tracing_subscriber::fmt::format; 

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

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Enable decoding.
    #[arg(long)]
    decode: bool,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: Option<String>,

    /// If set along with --decode, will use this prompt to initialize the decoder.
    #[arg(long)]
    decoder_prompt: Option<String>,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "false")]
    generate_profile: bool,

    #[arg(long)]
    output: Option<String>,
}

fn predict(prompt: String, model: &mut T5EncoderModel, cnn: &CNN, _profile_cnn: &CnnProfile, hashmap:&HashMap<String,usize>, device: &Device, profile: bool, buffer_file: &mut File, index_file: &mut File, buffer_file_seqs: &mut File, index_file_seqs: &mut File, embeds_file: &mut File, index: i32, curr_len: i32, seq_len: i32) -> Result<(Vec<String>, i32)>{
    let copy = &prompt;
    println!("{:?}", copy);
    // Replace each character in the string
    let replaced_values: Vec<Option<&usize>> = copy.chars()
        .map(|c| hashmap.get(&c.to_string()))
        .collect();
    
    let unknown_value: usize = 2; // Default value for None

    let ts: Vec<&usize> = replaced_values
        .iter()
        .map(|option| option.unwrap_or(&unknown_value))
        .collect();
    let mut tokens: Vec<&usize> = vec![hashmap.get("<AA2fold>").unwrap()];
    tokens.extend(ts.iter().clone());
    tokens.push(hashmap.get("</s>").unwrap());
    let tokens: Vec<i64> = tokens
    .iter()
    .map(|&num| *num as i64)
    .collect();

    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?.to_dtype(DType::U8)?;

    
    let ys = model.forward(&input_token_ids)?;

    let ys = ys.i((.., 1..ys.dims3()?.1-1))?;
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

    if profile{
        println!("generating profile");
        //let output = &cnn.forward(&ys)?;
        let probs: Tensor = softmax(output, 1)?;
        let lines: Vec<Vec<f32>> = probs.to_dtype(DType::F32)?.to_vec3()?[0].to_vec();

        let mut profile: Vec<f32> = Vec::new();
        for i in 0..lines[0].len(){
            for j in 0..20{
                profile.push(lines[j][i]);
                //print!("{:?} ", lines[j][i]);
            }
            //print!("\n");
        }

        
        let consensus = structure.clone();
        let mut pssm = compute_log_pssm(profile, structure.len())?;
        for pos in 0..consensus.len(){
            if consensus[pos]=='X'{
                for aa in 0..PROFILE_AA_SIZE{
                    pssm[pos * PROFILE_AA_SIZE as usize + aa as usize]=-1;
                }
            }
        }
        let result = to_buffer(consensus, structure.len(), pssm)?;
    
        //buffer_file.write_all(result_as_u8)?;
        
        buffer_file.write(&result)?;

        //write!(buffer_file, "{}",  &result)?;

        let index_content = format!("{}\t{}\t{}\n", index, curr_len, result.len());
        index_file.write(index_content.as_bytes())?;


        write!(buffer_file_seqs, "{}\n\0",  &prompt)?;
        
        let index_content = format!("{}\t{}\t{}\n", index, seq_len, &prompt.len()+2);
        index_file_seqs.write(index_content.as_bytes())?;

        let structure: String = structure.into_iter().collect();
        structures.push(structure.clone());
        Ok((structures, result.len() as i32))
    }

    else{
        let structure: String = structure.into_iter().collect();
        structures.push(structure.clone());

        Ok((structures, 0))
    }
            
}

fn process_fasta(input_path: &Path, output_path: &Path, model: &mut T5EncoderModel, cnn: &CNN, profile_cnn: &CnnProfile,   hashmap:&HashMap<String,usize>, device: &Device, profile: bool) -> io::Result<()> {
    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);

    let mut output_file = OpenOptions::new().create(true).write(true).truncate(true).open(output_path)?;
    let mut time_file = OpenOptions::new().create(true).write(true).truncate(true).open("./times.fasta")?;
    let mut buffer_file = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_ss")?;
    let mut index_file = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_ss.index")?;
    let mut dbtype_file = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_ss.dbtype")?;
    let mut buffer_file_seqs = OpenOptions::new().create(true).write(true).truncate(true).open("./profile")?;
    let mut index_file_seqs = OpenOptions::new().create(true).write(true).truncate(true).open("./profile.index")?;
    let mut dbtype_file_seqs = OpenOptions::new().create(true).write(true).truncate(true).open("./profile.dbtype")?;
    let mut buffer_file_h = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_h")?;
    let mut index_file_h = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_h.index")?;
    let mut dbtype_file_h = OpenOptions::new().create(true).write(true).truncate(true).open("./profile_h.dbtype")?;
    let mut lookup_file = OpenOptions::new().create(true).write(true).truncate(true).open("./profile.lookup")?;
    let mut embeds_file = OpenOptions::new().create(true).write(true).truncate(true).open("./embeds.txt")?;
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
            
            write!(buffer_file_h, "{}\n\0",  &l[1..])?;
            let index_content = format!("{}\t{}\t{}\n", index, header_len, l.len()+1);
            index_file_h.write(index_content.as_bytes())?;
            header_len += (l.len() + 1) as i32;
            let index_content = format!("{}\t{}\t{}\n", index, &l[1..], 0);
            lookup_file.write(index_content.as_bytes())?;
            if !s.is_empty() {                 
                let start_time = std::time::Instant::now();
                let prompt = s.clone();
                let prediction = predict(prompt, model, cnn, profile_cnn,  hashmap, device, profile, &mut buffer_file, &mut index_file,  &mut buffer_file_seqs, &mut index_file_seqs, &mut embeds_file, index, curr_len, seq_len);
                let prediction = prediction.unwrap();
                let res: Vec<String> = prediction.0;
                
                curr_len += prediction.1;
                seq_len += (s.len() + 2) as i32;
                println!("Took {:?}", start_time.elapsed());
                for mut value in res{
                    value.pop();
                    let _ = writeln!(output_file, "{}", value);
                    let _ = writeln!(time_file, "{} : {}", s.len(), start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 * 1e-9);
                }
                s.clear();
                
            }
            writeln!(output_file, "{}", line)?;
        }
        else {
            index += 1;
            s.push_str(&line);
        }
    }

    // Write the last sequence if not empty
    if !s.is_empty() {
        let prompt = s.clone();
        let prediction = predict(prompt, model, cnn, profile_cnn, hashmap, device, profile, &mut buffer_file, &mut index_file,  &mut buffer_file_seqs, &mut index_file_seqs, &mut embeds_file, index, curr_len, seq_len);
        let res: Vec<String> = prediction.unwrap().0;
        for value in &res{
            let _ = writeln!(output_file, "{}", value);
        }
    }

    Ok(())
}

fn compute_log_pssm(profile: Vec<f32>, query_length: usize) -> Result<Vec<i8>>{
   let pback: Vec<f32> = vec![0.0489372, 0.0306991, 0.101049, 0.0329671, 0.0276149, 0.0416262, 0.0452521, 0.030876, 0.0297251, 0.0607036, 0.0150238, 0.0215826, 0.0783843, 0.0512926, 0.0264886, 0.0610702, 0.0201311, 0.215998, 0.0310265, 0.0295417, 0.00001];
   //let pback: Vec<f32> = vec![0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.00001];
   let mut pssm: Vec<i8> = vec![0; query_length * PROFILE_AA_SIZE as usize];
   println!("{query_length}");
   println!("{}", profile.len());
   for pos in 0..query_length{
    for aa in 0..PROFILE_AA_SIZE{
        let aa_prob: f32 = profile[pos * PROFILE_AA_SIZE as usize + aa as usize];
        let idx: usize = pos  * PROFILE_AA_SIZE as usize+ aa as usize;
        let log_prob: f32 = (aa_prob/pback[aa as usize]).log2();
        let mut pssmval:f32 = log_prob * BIT_FACTOR + BIT_FACTOR * SCORE_BIAS;
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

fn to_buffer(consensus: Vec<char>, center_sequence_len: usize, pssm: Vec<i8>) -> Result<Vec<u8>>{
    let pssm_u8 = unsafe { 
        std::slice::from_raw_parts(pssm.as_ptr() as *const u8, pssm.len())
    };
    // for prob in pssm_u8 {
    //     print!("{prob}\n");
    // }
    let mut result: Vec<u8> = Vec::new();
    for pos in 0..center_sequence_len-1{
        for aa in 0..PROFILE_AA_SIZE{
            result.push(pssm_u8[pos*PROFILE_AA_SIZE as usize + aa as usize]);
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

pub struct CNN{
    conv1: Conv2d,
    // act: Activation,
    // dropout: Dropout,
    conv2: Conv2d
}

impl CNN{
    pub fn load(vb: VarBuilder, config: Conv2dConfig) -> Result<Self>{
        let conv1 = conv2d_non_square(1024, 32, 7, 1, config, vb.pp("classifier.0"))?;
        // let act = Activation::Relu;
        // let dropout = Dropout::new(0.0);
        let conv2 = conv2d_non_square(32, 20, 7, 1, config, vb.pp("classifier.3"))?;
        // Ok(Self { conv1, act, dropout, conv2 })
        Ok(Self { conv1, conv2 })
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
        Ok(xs)
    }
}

pub struct CnnProfile {
    conv1: Conv2d,
    // act: Activation,
    // dropout: Dropout,
    conv2: Conv2d
}

impl CnnProfile {
    pub fn load(vb: VarBuilder, config: Conv2dConfig) -> Result<Self>{
        let conv1 = conv2d_non_square(1024, 32, 7, 1, config, vb.pp("classifier.0"))?;
        // let act = Activation::Relu;
        // let dropout = Dropout::new(0.0);
        let conv2 = conv2d_non_square(32, 20, 7, 1, config, vb.pp("classifier.3"))?;
        // Ok(Self { conv1, act, dropout, conv2 })
        Ok(Self { conv1, conv2 })
    }

    #[allow(dead_code)]
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
        let xs: Tensor = xs.permute((0, 2, 1))?;
        let xs  = log_softmax(&xs, D::Minus1)?;
        Ok(xs)
    }
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
    cnn_filename: Vec<PathBuf>,
    profile_filename: Vec<PathBuf>
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<Self> {
        let device = device(args.cpu)?;
        let default_model = "t5-small".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        println!("{:?}", config_filename);
        //let tokenizer_filename = api.get("tokenizer.json")?;
        // let weights_filename = if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2"
        // {
        //     candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
        // } else {
        //     vec![api.get("model.safetensors")?]
        // };
        //println!("{:?}", weights_filename);
        let weights_filename = vec![PathBuf::from("/home/victor/.cache/huggingface/hub/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/model.safetensors")];
        let path = PathBuf::from("../cnn.safetensors");
        println!("{:?}", path);
        let profile_path = PathBuf::from("../new_cnn.safetensors");
        //let cnn_filename = path;
        let cnn_filename = vec![path];
        let profile_filename = vec![profile_path];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;

        Ok(
            Self {
                device,
                config,
                weights_filename,
                cnn_filename,
                profile_filename
            }
        )
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            //println!("3412423");
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        println!("{:?}", vb.contains_tensor("shared.weight"));
        //println!("3412423");
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
        Ok(CNN::load(vb,config)?)
    }

    pub fn build_profile_cnn(&self) -> Result<CnnProfile> {
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
        Ok(CnnProfile::load(vb,config)?)
    }

    // pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
    //     let vb = unsafe {
    //         VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
    //     };
    //     Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    // }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    //println!("1");
    let builder = T5ModelBuilder::load(&args)?;
    //println!("2");
    let device = &builder.device;
    let path = Path::new("/home/victor/git/candle/candle-examples/examples/t5/tokens.json");

    // Open the file in read-only mode
    let mut file = File::open(&path).expect("file not found");

    // Read the file content into a string
    let mut data = String::new();
    file.read_to_string(&mut data).expect("error reading the file");
    
    // Deserialize the JSON string into a HashMap
    let hashmap: HashMap<String, usize> = serde_json::from_str(&data)?;
    let mut model = builder.build_encoder()?;
    let cnn: &CNN = &builder.build_cnn()?;
    let profile_cnn = &builder.build_profile_cnn()?;
    match args.prompt {
        Some(prompt) => {
            let output = args.output.unwrap();
            let start = std::time::Instant::now();
            let _ = process_fasta(Path::new(&prompt), Path::new(&output), &mut model, cnn, profile_cnn, &hashmap, device, args.generate_profile);
            println!("Took {:?}", start.elapsed());
            //println!("finished");
            //let res = predict(prompt, &builder, &hashmap, device)?;
            //println!{"{:?}", res};   
        }
        None => {
        }
    }
    Ok(())
}
