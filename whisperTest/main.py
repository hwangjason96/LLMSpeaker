#################################
# Author: Jae Sung Hwang
# Last Modified: March 20th, 2026
# LLM Speaker OpenAI Whisper Test
#################################

import whisper

#making a function that will transcribe an audio to a variable.
#making type hints for both input parameter and the output return for my future self.
def transcribe(audio_path: str) -> str:
    #Loading whisper.
    print("Loading model...")
    model = whisper.load_model("base")

    #transcribe the audio to a text file
    print("Transcribing...")
    #Something to look out for, model.transcribe returns a dictionary of a lot of information.
    #we will only be using the text, portion for now, but later time stamps and other data values will be important.
    result = model.transcribe(audio_path, fp16=False)
    #notable thing, whisper will recognize pauses in speech and have different indexes for each time a thing is being said.

    #return the transcribed text
    return result["text"]

if __name__ == "__main__":
    text = transcribe("test.wav")
    #print the result
    print("You said: {}".format(text))
