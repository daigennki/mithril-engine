use crate::util::log_info;

pub struct GameContext {
    _pref_path: String
}
impl GameContext {
    // game context "constructor"
    pub fn new(log_file: &std::fs::File, pref_path: String) -> Result<GameContext, Box<dyn std::error::Error>> {
        log_info(log_file, "Constructing GameContext");



        Ok(GameContext { 
            _pref_path: pref_path
        })
    }

    pub fn render_loop(&self, log_file: &std::fs::File) -> Result<(), Box<dyn std::error::Error>> {
        log_info(log_file, "Render loop goes here");
        Ok(())
    }
}