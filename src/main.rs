/* ----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
------------------------------------------------------------------------------ */
extern crate sdl2;

fn main() {
    let org_name = "daigennki";
    let game_name = "MithrilEngine";

    // get command line arguments
    let args: Vec<String> = std::env::args().collect();

    let prefPathResult = sdl2::filesystem::pref_path(org_name, game_name);
    let prefPath = "";
    match prefPathResult {
        Ok(s) => {
            panic!("Using preferences path: {}", s);
            //prefPath = s;
        },
        Err(e) => panic!("Failed to get preferences path! The application might not have permission to it.")
    }
}
