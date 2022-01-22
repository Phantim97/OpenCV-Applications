#pragma once
enum class VidcapMode
{
	VIDEO,
	LIVE
};

void face_swap_main();
void face_swap_video(VidcapMode vm);