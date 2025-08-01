from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import os
import wikipedia
import random
import textwrap
import math
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import whisper
from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    ImageClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip
)

# Set language if needed
wikipedia.set_lang("en")

def render_text_with_highlighted_word(
    chunk_words,
    highlight_index,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    fontsize=30,
    max_width=1000,
    padding=10,
    text_color=(255, 255, 255),
    highlight_color=(255, 0, 0),
    highlight_text_color=(255, 255, 255),
    bg_opacity=200,
    duration=1.0,
    position=("center", "center"),
):

    font = ImageFont.truetype(font_path, fontsize)
    space_width = font.getlength(" ")

    # Accurate line height
    ascent, descent = font.getmetrics()
    line_height = ascent + descent

    # Manual line wrapping
    lines = []
    current_line = []
    current_width = 0

    for word in chunk_words:
        word_width = font.getlength(word)
        if current_width + word_width + space_width * len(current_line) <= max_width - 2 * padding:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(current_line)
            current_line = [word]
            current_width = word_width
    if current_line:
        lines.append(current_line)

    # Compute image height and width
    img_width = max_width
    img_height = (line_height + padding) * len(lines) + padding

    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    word_counter = 0
    for i, line in enumerate(lines):
        y = padding + i * line_height

        # Compute total line width (words + spaces)
        line_word_widths = [font.getlength(w.upper() if word_counter + idx == highlight_index else w.lower())
                            for idx, w in enumerate(line)]
        total_line_width = sum(line_word_widths) + space_width * (len(line) - 1)

        x = (img_width - total_line_width) / 2

        for j, word in enumerate(line):
            display_word = word.upper() if word_counter == highlight_index else word.lower()
            color = highlight_text_color if word_counter == highlight_index else text_color

            word_width = font.getlength(display_word)

            if word_counter == highlight_index:
                bbox = draw.textbbox((x, y), display_word, font=font)
                box_x1, box_y1, box_x2, box_y2 = bbox
                border_radius = 12
                draw.rounded_rectangle(
                    [box_x1 - 5, box_y1 - 5, box_x2 + 5, box_y2 + 5],
                    radius=border_radius,
                    fill=(*highlight_color, bg_opacity)
                )

            draw.text((x, y), display_word, font=font, fill=color)
            x += word_width + space_width
            word_counter += 1


    return (
        ImageClip(np.array(img), duration=duration)
        .set_position(position)
    )


def build_timing_plan(words, total_duration):
    """Temporary timing plan that evenly spaces words across total_duration."""
    avg_duration = total_duration / len(words)
    return [
        {"word": word, "start": i * avg_duration, "duration": avg_duration}
        for i, word in enumerate(words)
    ]

def build_timing_plan_with_whisper(audio_path, reference_text):
    model = whisper.load_model("base")  # Use "tiny", "base", "small", "medium", or "large"
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

    # Attempt to align words using the provided reference_text
    words = []
    for segment in result.get("segments", []):
        for word_obj in segment.get("words", []):
            word = word_obj.get("word", "").strip()
            start = word_obj.get("start")
            end = word_obj.get("end")
            if start is not None and end is not None:
                words.append({
                    "word": word,
                    "start": start,
                    "duration": end - start
                })
    return words


def get_random_wiki_content():
    # Get random article title
    title = wikipedia.random()
    # Fetch full page
    page = wikipedia.page(title)
    return page,title

def main():
    # Get wiki text
    try:
        page,title = get_random_wiki_content()
    except:
        page,title = get_random_wiki_content()

    # Get specific page in dev
    # title = "Autopoiesis and Cognition: The Realization of the Living"
    # page = wikipedia.page(title)

    page_text = page.summary ## Limited for development
    ## Consider saving the summary as a txt file

    print(f"✅ Wiki article loaded: {title}")

    # Choose model
    model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
    # model_name = "tts_models/en/jenny/jenny"

    # Download model & config
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model(model_name)

    vocoder_name = model_item["default_vocoder"]
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    # Initialize synthesizer (CPU mode)
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        use_cuda=False
    )

    # Text input
    # text = "This is a test of Coqui TTS running entirely on CPU."
    text = page_text

    # Run TTS
    wav = synthesizer.tts(text)
    # Save output
    # CREATE FOLDERS FOR EACH VIDEO LATER os.mkdir(title.replace(' ','_'))
    save_folder_path = "Wiki_Brainrot_"+title.replace(' ','_')
    os.mkdir(save_folder_path)
    output_path = save_folder_path+"/Wiki_TTS_"+title.replace(' ','_')+".wav"
    synthesizer.save_wav(wav, output_path)

    print(f"✅ Audio saved at {output_path}")

    print(f"🔃 Started creating the video...")

    # ----- Paths -----
    video_path = "/mnt/c/Users/dorgoz/Downloads/minecraftparrkour_cropped.mp4"
    tts_audio_path = output_path
    bg_music_path = "/mnt/c/Users/dorgoz/Downloads/Alla-Turca.mp3"
    output_video_path = save_folder_path+"/Wiki_Brainrot_"+title.replace(' ','_')+"_video.mp4"
    transcript_text = text

   # ----- Load media -----
    tts_audio = AudioFileClip(tts_audio_path)
    tts_duration = tts_audio.duration

    # Load video with a random offset for variation
    # Ensure we don’t exceed total video length
    video_duration = VideoFileClip(video_path).duration
    max_offset = max(0, video_duration - tts_duration, 0)
    offset = random.uniform(0, min(240, max_offset))
    video = VideoFileClip(video_path).subclip(offset, offset + tts_duration)
    music = AudioFileClip(bg_music_path).subclip(0, tts_duration).volumex(0.3)


    # ----- Mix TTS and background music -----
    combined_audio = CompositeAudioClip([music, tts_audio])
    # video = mp.vfx.set_audio(video,combined_audio)
    video = video.set_audio(combined_audio)


    # ----- Create timed text overlays -----
    # target_width = video.w
    # target_height = video.h
    chunk_size = 6  # words per chunk
    words = transcript_text.split()
    # timing_plan = build_timing_plan(words, tts_duration)
    timing_plan = build_timing_plan_with_whisper(tts_audio_path, transcript_text)
    words = [item["word"] for item in timing_plan]

    clips = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        plan_chunk = timing_plan[i:i + chunk_size]

        for j, item in enumerate(plan_chunk):
            txt_clip = render_text_with_highlighted_word(
                chunk_words=chunk_words,
                highlight_index=j,
                font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                fontsize=80,
                max_width=video.w - 80,
                duration=item["duration"]
            ).set_start(item["start"])

            clips.append(txt_clip)

    # ----- Combine everything -----
    final = CompositeVideoClip([video] + clips) # used to be highlighted_clips
    final = final.set_duration(tts_duration)

    # ----- Export -----
    final.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=30)
    print(f"✅ Video saved at: {output_video_path}")


def generate_video_from_text(intro_title, intro_text):
    title = intro_title
    text = intro_text

    print(f"✅ Wiki article loaded: {title}")

    # Choose model
    model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
    # model_name = "tts_models/en/jenny/jenny"

    # Download model & config
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model(model_name)

    vocoder_name = model_item["default_vocoder"]
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    # Initialize synthesizer (CPU mode)
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        use_cuda=False
    )

    # Run TTS
    wav = synthesizer.tts(text)
    # Save output
    # CREATE FOLDERS FOR EACH VIDEO LATER os.mkdir(title.replace(' ','_'))
    save_folder_path = "Wiki_Brainrot_"+title.replace(' ','_')
    os.mkdir(save_folder_path)
    output_path = save_folder_path+"/Wiki_TTS_"+title.replace(' ','_')+".wav"
    synthesizer.save_wav(wav, output_path)

    print(f"✅ Audio saved at {output_path}")

    print(f"🔃 Started creating the video...")

    # ----- Paths -----
    video_path = "/mnt/c/Users/dorgoz/Downloads/minecraftparrkour_cropped.mp4"
    tts_audio_path = output_path
    bg_music_path = "/mnt/c/Users/dorgoz/Downloads/Alla-Turca.mp3"
    output_video_path = save_folder_path+"/Wiki_Brainrot_"+title.replace(' ','_')+"_video.mp4"
    transcript_text = text

   # ----- Load media -----
    tts_audio = AudioFileClip(tts_audio_path)
    tts_duration = tts_audio.duration

    # Load video with a random offset for variation
    # Ensure we don’t exceed total video length
    video_duration = VideoFileClip(video_path).duration
    max_offset = max(0, video_duration - tts_duration, 0)
    offset = random.uniform(0, min(240, max_offset))
    video = VideoFileClip(video_path).subclip(offset, offset + tts_duration)
    music = AudioFileClip(bg_music_path).subclip(0, tts_duration).volumex(0.3)


    # ----- Mix TTS and background music -----
    combined_audio = CompositeAudioClip([music, tts_audio])
    # video = mp.vfx.set_audio(video,combined_audio)
    video = video.set_audio(combined_audio)


    # ----- Create timed text overlays -----
    # target_width = video.w
    # target_height = video.h
    chunk_size = 6  # words per chunk
    words = transcript_text.split()
    # timing_plan = build_timing_plan(words, tts_duration)
    timing_plan = build_timing_plan_with_whisper(tts_audio_path, transcript_text)
    words = [item["word"] for item in timing_plan]

    clips = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        plan_chunk = timing_plan[i:i + chunk_size]

        for j, item in enumerate(plan_chunk):
            txt_clip = render_text_with_highlighted_word(
                chunk_words=chunk_words,
                highlight_index=j,
                font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                fontsize=80,
                max_width=video.w - 80,
                duration=item["duration"]
            ).set_start(item["start"])

            clips.append(txt_clip)

    # ----- Combine everything -----
    final = CompositeVideoClip([video] + clips) # used to be highlighted_clips
    final = final.set_duration(tts_duration)

    # ----- Export -----
    final.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=30)
    print(f"✅ Video saved at: {output_video_path}")

if __name__ == "__main__":
    main()

