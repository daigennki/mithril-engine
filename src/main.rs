/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
----------------------------------------------------------------------------- */
mod gamecontext;
mod util;

fn main() 
{
        let org_name = "daigennki";
        let game_name = "MithrilEngine";

        // get preferences path
        // (log, config, and save data files will be saved here)
        let pref_path_str;
        match sdl2::filesystem::pref_path(org_name, game_name) {
                Ok(s) => pref_path_str = s,
                Err(e) => {
                        let es = e.to_string();
                        let e_fmt = format!(
				"Failed to get preferences path: {}", 
				&es
			);
                        print_error_unlogged(&e_fmt);
                        return;
                }
        }
        println!("Using preferences path: {}", &pref_path_str);

        // open log file
        let log_file_path = [&pref_path_str, "game.log"].concat();
        let log_file: std::fs::File;
        match std::fs::File::create(log_file_path) {
                Ok(f) => log_file = f,
                Err(e) => {
                        let e_str = e.to_string();
                        let e_fmt = format!(
				"Failed to create log file: {}", 
				&e_str
			);
                        print_error_unlogged(&e_fmt);
                        return;
                }
        }

        // construct GameContext
        let gctx_res = gamecontext::GameContext::new(log_file, pref_path_str);
        let gctx;
        match gctx_res {
                Ok(g) => gctx = g,
                Err(()) => return       // error logging is handled inside `new`
        }

        // run render loop
        gctx.render_loop();
}

fn print_error_unlogged(s: &str) 
{
        println!("{}", &s);
        match msgbox::create(
                "MithrilEngine Error", 
                &s, 
                msgbox::common::IconType::Error
        ) {
                Ok(r) => r,
                Err(mbe) => println!(
			"msgbox::create failed: {}", 
			&mbe.to_string()
		)
        }
}
