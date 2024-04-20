use std::ffi::{CStr, CString, c_char};
use crate::ProstT5;

#[no_mangle]
pub extern "C" fn prostt5_free(ptr: *mut ProstT5) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

#[no_mangle]
pub extern "C" fn prostt5_load(base_path: *const c_char, profile: bool, cpu: bool, cache: bool) -> *mut ProstT5 {
    let path = unsafe { CStr::from_ptr(base_path).to_str().unwrap() };
    match ProstT5::load(path.to_string(), profile, cpu, cache) {
        Ok(prostt5) => Box::into_raw(Box::new(prostt5)),
        Err(_) => std::ptr::null_mut()
    }
}

#[no_mangle]
pub extern "C" fn prostt5_predict(ptr: *mut ProstT5, sequence: *const c_char) -> *const c_char {
    let seq = unsafe { CStr::from_ptr(sequence).to_str().unwrap() };
    let prostt5 = unsafe { &mut *ptr };
    let result = prostt5.predict(seq.to_string());

    match result {
        Ok((res, _status_code)) => {
            let c_str = CString::new(res).expect("CString::new failed");
            c_str.into_raw()
        },
        Err(_) => std::ptr::null()
    }
}

