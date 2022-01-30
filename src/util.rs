use std::io::Write;

pub fn log_info(mut log_file: &std::fs::File, s: &str) 
{
	println!("{}", s);
	write!(log_file, "{}\n", s)
		.unwrap_or_else(|e| println!("log_info failed to print to log file: {}", e));
}
