/* ----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
------------------------------------------------------------------------------ */

fn log_info(s: &str) {
    println!("{}", s);
    // TODO: print to log file
}
fn log_error(e: &str) {
    let formatted_log_str = ["ERROR: ", &e].concat();
    log_info(&formatted_log_str);
    let box_result = msgbox::create("MithrilEngine Error", e, msgbox::common::IconType::Error);
    match box_result {
        Ok(r) => r,
        Err(mbe) => {
            let mbe_str = ["Error occurred while trying to create error message box: ", &mbe.to_string()].concat();
            log_info(&mbe_str);
        }
    }
}

fn main() {
    let org_name = "daigennki";
    let game_name = "MithrilEngine";

    // get command line arguments
    // let args: Vec<String> = std::env::args().collect();

    // get preferences path (config and save data files will be saved here)
    let pref_path_result = sdl2::filesystem::pref_path(org_name, game_name);
    //let mut pref_path_str = String::new();
    match pref_path_result {
        Ok(s) => {
            let info_str = ["Using preferences path: ", &s].concat();
            log_info(&info_str);
            //pref_path_str = s;
        },
        Err(e) => {
            let error_str = ["Failed to get preferences path: ", &e.to_string()].concat();
            log_error(&error_str);
            return;
        }
    }
}
