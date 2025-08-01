# WikiRot

## What?
This little pipeline pulls a random Wikipedia article and creates a text to speech dictation of it using coqui-tts.
It then loads a long video, chops it to the length of the audio file, adds the text of the speech (at correct timestamps) to the video, and saves the corresponding video file.

TLDR: Free TikTok Brainrot Generator

|![[/readmeimages/exp1.png|150]]|![[/readmeimages/exp2.png|150]]|![[/readmeimages/exp3.png|150]]

--- 

## Dependencies
You will need:
- Windows Subsystem for Linux (WSL)
- A video file longer than your TTS generation (or several, will add options to choose (randomly or not))
- Crop the video for mobile aspect ratio (well do this after installing ffmpeg(see Step2) of course):
```bash
ffmpeg -i /path/video.mp4 -vf "crop=in_h*9/16:in_h,scale=1080:1920" /path/video_cropped.mp4
```
- An ambient music file longer than your TTS generation (or several, will add options to choose (randomly or not))

---
## Setup

- Step1: Open WSL
```bash
# Create folder for the project
mkdir WikiTTS
# Move to the directory
cd WikiTTS
```
- Step 2: Install dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip ffmpeg espeak
sudo apt install imagemagick
```
- Sub-step: Edit the ImageMagick security policy file:
```bash 
sudo nano /etc/ImageMagick-6/policy.xml
```
change <policy domain="path" rights="none" pattern="@*"/> `none` to `read|write`
```xml
<policy domain="path" rights="none" pattern="@*"/>
<policy domain="coder" rights="none" pattern="PDF" />
```
- Step 3: Create virtual environment
```bash
python3 -m venv tts_env
source tts_env/bin/activate
```
- Step 4: Install packages
```bash
pip install numpy
pip install coqui-tts
pip install moviepy==1.0.3
pip install openai-whisper
pip install wikipedia
pip install wikipedia-api
```
---
## Usage
- Step 1: Open WSL from the terminal app
- Step 2: Move to the project path and activate the virtual environment
``` bash
# Move to the directory
cd WikiTTS
# activate environment
source tts_env/bin/activate
```
- Step 3: Make sure the python script is at this directory
- Step 4: Set the paths of your resources (video and bg music) in the python script
- Step 5: Run python script to generate video from random article
```bash
# run python script
python3 movietts.py
```

## ToDo

- [ ] Web-based GUI - maybe with streamlit?
- [x] Fix word highlight timestamps
- [ ] Different TTS models
- [ ] Multiple BG music options
- [ ] Multiple video options
- [ ] Parallel batch processing?
- [ ] AI bg video generation? (overkill imo)
- [ ] AI bg music generation? (overkill imo)
- [ ] Auto-upload draft to TikTok
- [ ] Auto-upload draft to other social media?
- [ ] Auto-publish?
- [ ] Auto collect money?
- [ ] Auto spend money on games and beer
- [ ] Auto write passive income pipeline because this one is spent on games and beer
---

---
