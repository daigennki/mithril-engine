use std::io::Write;

pub fn log_info(mut log_file: &std::fs::File, s: &str) 
{
	println!("{}", s);
	let str_with_newline = format!("{}\n", s);
	match log_file.write_all(str_with_newline.as_bytes()) {
		Ok(()) => (),
		Err(e) => println!("log_info failed to print to log file: {}", e)
	}
}
