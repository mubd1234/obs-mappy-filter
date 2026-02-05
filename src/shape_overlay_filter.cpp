#include "shape_overlay_filter.h"

#include <util/platform.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <mutex>
#include <string>

#define BLOG_CHANNEL "shape-overlay"

struct shape_overlay_filter_data {
	obs_source_t *source;
	std::mutex mutex;

	std::string template_path;
	std::string overlay_path;

	cv::Mat template_gray;
	cv::Mat overlay_bgra;
	cv::Mat overlay_draw;

	float threshold = 0.8f;
	uint32_t interval_ms = 100;
	float opacity = 1.0f;
	int offset_x = 0;
	int offset_y = 0;
	bool scale_overlay = true;
	bool only_when_matched = true;

	uint64_t last_detect_ts = 0;
	int last_x = 0;
	int last_y = 0;
	float last_score = 0.0f;
	bool last_valid = false;
	bool warned_format = false;
};

static const char *shape_overlay_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("ShapeOverlayFilter");
}

static cv::Mat load_template_gray(const std::string &path)
{
	if (path.empty()) {
		return cv::Mat();
	}

	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	return img;
}

static cv::Mat load_overlay_bgra(const std::string &path)
{
	if (path.empty()) {
		return cv::Mat();
	}

	cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (img.empty()) {
		return img;
	}

	if (img.channels() == 4) {
		return img;
	}

	cv::Mat converted;
	if (img.channels() == 3) {
		cv::cvtColor(img, converted, cv::COLOR_BGR2BGRA);
	} else if (img.channels() == 1) {
		cv::cvtColor(img, converted, cv::COLOR_GRAY2BGRA);
	} else {
		return cv::Mat();
	}

	return converted;
}

static void shape_overlay_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_double(settings, "threshold", 0.8);
	obs_data_set_default_int(settings, "interval_ms", 100);
	obs_data_set_default_double(settings, "opacity", 100.0);
	obs_data_set_default_int(settings, "offset_x", 0);
	obs_data_set_default_int(settings, "offset_y", 0);
	obs_data_set_default_bool(settings, "scale_overlay", true);
	obs_data_set_default_bool(settings, "only_when_matched", true);
}

static obs_properties_t *shape_overlay_filter_properties(void *unused)
{
	UNUSED_PARAMETER(unused);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_path(props, "template_path", obs_module_text("TemplatePath"),
				OBS_PATH_FILE, "PNG files (*.png)", NULL);
	obs_properties_add_path(props, "overlay_path", obs_module_text("OverlayPath"),
				OBS_PATH_FILE, "PNG files (*.png)", NULL);

	obs_properties_add_float_slider(props, "threshold",
				obs_module_text("Threshold"), 0.0, 1.0, 0.01);
	obs_properties_add_int(props, "interval_ms",
				obs_module_text("IntervalMs"), 0, 2000, 10);
	obs_properties_add_float_slider(props, "opacity",
				obs_module_text("Opacity"), 0.0, 100.0, 1.0);
	obs_properties_add_int(props, "offset_x",
				obs_module_text("OffsetX"), -4096, 4096, 1);
	obs_properties_add_int(props, "offset_y",
				obs_module_text("OffsetY"), -4096, 4096, 1);
	obs_properties_add_bool(props, "scale_overlay",
				obs_module_text("ScaleToTemplate"));
	obs_properties_add_bool(props, "only_when_matched",
				obs_module_text("OnlyWhenMatched"));

	return props;
}

static void shape_overlay_filter_update(void *data, obs_data_t *settings)
{
	shape_overlay_filter_data *filter = static_cast<shape_overlay_filter_data *>(data);

	std::lock_guard<std::mutex> lock(filter->mutex);

	filter->template_path = obs_data_get_string(settings, "template_path");
	filter->overlay_path = obs_data_get_string(settings, "overlay_path");
	filter->threshold = static_cast<float>(obs_data_get_double(settings, "threshold"));
	filter->interval_ms = static_cast<uint32_t>(obs_data_get_int(settings, "interval_ms"));
	filter->opacity = static_cast<float>(obs_data_get_double(settings, "opacity") / 100.0);
	filter->offset_x = static_cast<int>(obs_data_get_int(settings, "offset_x"));
	filter->offset_y = static_cast<int>(obs_data_get_int(settings, "offset_y"));
	filter->scale_overlay = obs_data_get_bool(settings, "scale_overlay");
	filter->only_when_matched = obs_data_get_bool(settings, "only_when_matched");

	filter->opacity = std::clamp(filter->opacity, 0.0f, 1.0f);
	filter->threshold = std::clamp(filter->threshold, 0.0f, 1.0f);

	filter->template_gray = load_template_gray(filter->template_path);
	filter->overlay_bgra = load_overlay_bgra(filter->overlay_path);

	if (!filter->overlay_bgra.empty() && filter->scale_overlay && !filter->template_gray.empty()) {
		cv::resize(filter->overlay_bgra, filter->overlay_draw,
				cv::Size(filter->template_gray.cols, filter->template_gray.rows),
				0.0, 0.0, cv::INTER_AREA);
	} else {
		filter->overlay_draw = filter->overlay_bgra;
	}

	filter->last_valid = false;
}

static void *shape_overlay_filter_create(obs_data_t *settings, obs_source_t *source)
{
	shape_overlay_filter_data *filter = new shape_overlay_filter_data();
	filter->source = source;

	shape_overlay_filter_update(filter, settings);
	return filter;
}

static void shape_overlay_filter_destroy(void *data)
{
	shape_overlay_filter_data *filter = static_cast<shape_overlay_filter_data *>(data);
	delete filter;
}

static bool detect_template(const cv::Mat &frame_gray, const cv::Mat &templ_gray,
		float threshold, int *out_x, int *out_y, float *out_score)
{
	if (frame_gray.empty() || templ_gray.empty()) {
		return false;
	}

	if (templ_gray.cols > frame_gray.cols || templ_gray.rows > frame_gray.rows) {
		return false;
	}

	cv::Mat result;
	cv::matchTemplate(frame_gray, templ_gray, result, cv::TM_CCOEFF_NORMED);

	double min_val = 0.0;
	double max_val = 0.0;
	cv::Point min_loc;
	cv::Point max_loc;
	cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

	if (out_score) {
		*out_score = static_cast<float>(max_val);
	}

	if (max_val >= threshold) {
		if (out_x) {
			*out_x = max_loc.x;
		}
		if (out_y) {
			*out_y = max_loc.y;
		}
		return true;
	}

	return false;
}

static void blend_overlay_bgra(uint8_t *dst, uint32_t dst_linesize,
		int frame_w, int frame_h, const cv::Mat &overlay,
		int dst_x, int dst_y, float opacity)
{
	if (overlay.empty()) {
		return;
	}

	const int overlay_w = overlay.cols;
	const int overlay_h = overlay.rows;

	int start_x = std::max(0, dst_x);
	int start_y = std::max(0, dst_y);
	int end_x = std::min(frame_w, dst_x + overlay_w);
	int end_y = std::min(frame_h, dst_y + overlay_h);

	if (start_x >= end_x || start_y >= end_y) {
		return;
	}

	const int overlay_x0 = start_x - dst_x;
	const int overlay_y0 = start_y - dst_y;

	for (int oy = overlay_y0, fy = start_y; fy < end_y; ++fy, ++oy) {
		const uint8_t *src_row = overlay.ptr<uint8_t>(oy);
		uint8_t *dst_row = dst + (static_cast<size_t>(fy) * dst_linesize);

		for (int ox = overlay_x0, fx = start_x; fx < end_x; ++fx, ++ox) {
			const uint8_t *src_px = src_row + (static_cast<size_t>(ox) * 4u);
			uint8_t *dst_px = dst_row + (static_cast<size_t>(fx) * 4u);

			const int src_alpha = static_cast<int>(static_cast<float>(src_px[3]) * opacity + 0.5f);
			if (src_alpha <= 0) {
				continue;
			}

			const int inv_alpha = 255 - src_alpha;
			dst_px[0] = static_cast<uint8_t>((src_px[0] * src_alpha + dst_px[0] * inv_alpha + 127) / 255);
			dst_px[1] = static_cast<uint8_t>((src_px[1] * src_alpha + dst_px[1] * inv_alpha + 127) / 255);
			dst_px[2] = static_cast<uint8_t>((src_px[2] * src_alpha + dst_px[2] * inv_alpha + 127) / 255);
			dst_px[3] = 255;
		}
	}
}

static obs_source_frame *shape_overlay_filter_video(void *data, obs_source_frame *frame)
{
	if (!frame) {
		return nullptr;
	}

	shape_overlay_filter_data *filter = static_cast<shape_overlay_filter_data *>(data);

	if (frame->format != VIDEO_FORMAT_BGRA && frame->format != VIDEO_FORMAT_BGRX) {
		if (!filter->warned_format) {
			blog(LOG_WARNING, "[%s] Unsupported frame format: %d (expected BGRA/BGRX)",
				BLOG_CHANNEL, frame->format);
			filter->warned_format = true;
		}
		return frame;
	}

	cv::Mat template_gray;
	cv::Mat overlay_draw;
	float threshold = 0.0f;
	float opacity = 1.0f;
	uint32_t interval_ms = 0;
	int offset_x = 0;
	int offset_y = 0;
	bool only_when_matched = true;

	uint64_t last_detect_ts = 0;
	int last_x = 0;
	int last_y = 0;
	bool last_valid = false;
	float last_score = 0.0f;

	{
		std::lock_guard<std::mutex> lock(filter->mutex);
		template_gray = filter->template_gray;
		overlay_draw = filter->overlay_draw;
		threshold = filter->threshold;
		opacity = filter->opacity;
		interval_ms = filter->interval_ms;
		offset_x = filter->offset_x;
		offset_y = filter->offset_y;
		only_when_matched = filter->only_when_matched;

		last_detect_ts = filter->last_detect_ts;
		last_x = filter->last_x;
		last_y = filter->last_y;
		last_valid = filter->last_valid;
		last_score = filter->last_score;
	}

	if (template_gray.empty() || overlay_draw.empty()) {
		return frame;
	}

	const uint64_t now = os_gettime_ns();
	const uint64_t interval_ns = static_cast<uint64_t>(interval_ms) * 1000000ull;
	const bool should_detect = (interval_ms == 0) || (now - last_detect_ts >= interval_ns);
	bool state_updated = false;

	if (should_detect) {
		cv::Mat frame_bgra(frame->height, frame->width, CV_8UC4, frame->data[0], frame->linesize[0]);
		cv::Mat frame_gray;
		cv::cvtColor(frame_bgra, frame_gray, cv::COLOR_BGRA2GRAY);

		float score = 0.0f;
		int found_x = 0;
		int found_y = 0;
		bool matched = detect_template(frame_gray, template_gray, threshold,
				&found_x, &found_y, &score);

		last_score = score;
		if (matched) {
			last_x = found_x;
			last_y = found_y;
			last_valid = true;
		} else if (only_when_matched) {
			last_valid = false;
		}

		last_detect_ts = now;
		state_updated = true;
	}

	if (state_updated) {
		std::lock_guard<std::mutex> lock(filter->mutex);
		filter->last_detect_ts = last_detect_ts;
		filter->last_x = last_x;
		filter->last_y = last_y;
		filter->last_valid = last_valid;
		filter->last_score = last_score;
	}

	if (!last_valid) {
		return frame;
	}

	const int draw_x = last_x + offset_x;
	const int draw_y = last_y + offset_y;

	blend_overlay_bgra(frame->data[0], frame->linesize[0],
			frame->width, frame->height,
			overlay_draw, draw_x, draw_y, opacity);

	return frame;
}

struct obs_source_info shape_overlay_filter = {
	.id = "shape_overlay_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC_VIDEO,
	.get_name = shape_overlay_filter_get_name,
	.create = shape_overlay_filter_create,
	.destroy = shape_overlay_filter_destroy,
	.get_defaults = shape_overlay_filter_defaults,
	.get_properties = shape_overlay_filter_properties,
	.update = shape_overlay_filter_update,
	.filter_video = shape_overlay_filter_video,
};
