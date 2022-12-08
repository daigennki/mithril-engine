use proc_macro::{ self, TokenStream };
use quote::quote;
use syn::{ parse_macro_input, DeriveInput };

#[proc_macro_derive(EntityComponent)]
pub fn derive_entity_component(input: TokenStream) -> TokenStream 
{
	let DeriveInput { ident, .. } = parse_macro_input!(input);
	let output = quote! {
		#[typetag::deserialize]
		impl EntityComponent for #ident
		{
			fn add_to_entity(self: Box<Self>, world: &mut shipyard::World, eid: shipyard::EntityId)
			{
				world.add_component(eid, (*self,));
			}
		}
	};
	output.into()
}

#[proc_macro_derive(UniqueComponent)]
pub fn derive_unique_component(input: TokenStream) -> TokenStream 
{
	let DeriveInput { ident, .. } = parse_macro_input!(input);
	let output = quote! {
		#[typetag::deserialize]
		impl UniqueComponent for #ident
		{
			fn add_to_world(self: Box<Self>, world: &mut shipyard::World)
			{
				world.add_unique(*self);
			}
		}
	};
	output.into()
}

