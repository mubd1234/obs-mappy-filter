#include <obs-module.h>
#include "shape_overlay_filter.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-shape-overlay", "en-US")

bool obs_module_load(void)
{
	obs_register_source(&shape_overlay_filter);
	return true;
}

const char *obs_module_description(void)
{
	return "Template match shape overlay filter";
}
