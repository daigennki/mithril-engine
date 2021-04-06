/* ----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
------------------------------------------------------------------------------ */
use std::io::Write;

fn log_info(mut log_file: &std::fs::File, s: &str) {
    println!("{}", &s);
    let str_with_newline = format!("{}\n", s);
    match log_file.write_all(str_with_newline.as_bytes()) {
        Ok(()) => (),
        Err(e) => println!("log_info failed to print to log file: {}", e.to_string())
    }
}
fn print_error_unlogged(s: &str) {
    println!("{}", &s);
    match msgbox::create("MithrilEngine Error", &s, msgbox::common::IconType::Error) {
        Ok(r) => r,
        Err(mbe) => println!("msgbox::create failed: {}", &mbe.to_string())
    }
}

fn logged_main(/*log_file: &std::fs::File*/) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn main() {
    let org_name = "daigennki";
    let game_name = "MithrilEngine";

    // get command line arguments
    // let args: Vec<String> = std::env::args().collect();

    // get preferences path (log, config, and save data files will be saved here)
    let pref_path_str;
    match sdl2::filesystem::pref_path(org_name, game_name) {
        Ok(s) => {
            println!("Using preferences path: {}", &s);
            pref_path_str = s;
        },
        Err(e) => {
            let e_str = format!("Failed to get preferences path: {}", &e.to_string());
            print_error_unlogged(&e_str);
            return;
        }
    }

    // open log file
    let log_file_path = [&pref_path_str, "game.log"].concat();
    let log_file: std::fs::File;
    match std::fs::File::create(log_file_path) {
        Ok(f) => log_file = f,
        Err(e) => {
            let e_str = format!("Failed to create log file: {}", &e.to_string());
            print_error_unlogged(&e_str);
            return;
        }
    }

    // print start date and time
    log_info(&log_file, &format!("--- INIT {} ---", chrono::Local::now().to_rfc2822()));

    match logged_main(/*&log_file*/) {
        Ok(()) => (),
        Err(e) => {
            log_info(&log_file, &format!("ERROR: {}", &e.to_string()));
            match msgbox::create("MithrilEngine Error", &e.to_string(), msgbox::common::IconType::Error) {
                Ok(r) => r,
                Err(mbe) => {
                    let mbe_str = ["Error occurred while trying to create error message box: ", &mbe.to_string()].concat();
                    log_info(&log_file, &mbe_str);
                }
            }
        }
    }

    // print end date and time
    log_info(&log_file, &format!("--- EXIT {} ---", chrono::Local::now().to_rfc2822()));
}
