/* ----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
------------------------------------------------------------------------------ */
mod gamecontext;
mod util;

fn print_error_unlogged(s: &str) {
    println!("{}", &s);
    match msgbox::create("MithrilEngine Error", &s, msgbox::common::IconType::Error) {
        Ok(r) => r,
        Err(mbe) => println!("msgbox::create failed: {}", &mbe.to_string())
    }
}

fn logged_main(log_file: &std::fs::File, pref_path_str: String) -> Result<(), Box<dyn std::error::Error>> {
    // initialization
    let gctx = gamecontext::GameContext::new(log_file, pref_path_str)?;

    // render loop
    gctx.render_loop(log_file)?;
    
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
    util::log_info(&log_file, &format!("--- INIT {} ---", chrono::Local::now().to_rfc3339()));

    // construct and run GameContext
    match logged_main(&log_file, pref_path_str) {
        Ok(()) => (),
        Err(e) => {
            util::log_info(&log_file, &format!("ERROR: {}", &e.to_string()));
            match msgbox::create("MithrilEngine Error", &e.to_string(), msgbox::common::IconType::Error) {
                Ok(r) => r,
                Err(mbe) => {
                    let mbe_str = ["Error occurred while trying to create error message box: ", &mbe.to_string()].concat();
                    util::log_info(&log_file, &mbe_str);
                }
            }
        }
    }

    // print end date and time
    util::log_info(&log_file, &format!("--- EXIT {} ---", chrono::Local::now().to_rfc3339()));
}
