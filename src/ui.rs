/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use super::rendercontext;

pub struct Canvas
{
    ui_pipeline: rendercontext::pipeline::Pipeline
}
impl Canvas
{
    pub fn new(render_context: &rendercontext::RenderContext) -> Result<Canvas, Box<dyn std::error::Error>>
    {
        let ui_pipeline = render_context.create_pipeline_for_swapchain("ui.yaml")?;
        
        Ok(Canvas{
            ui_pipeline: ui_pipeline
        })
    }
}