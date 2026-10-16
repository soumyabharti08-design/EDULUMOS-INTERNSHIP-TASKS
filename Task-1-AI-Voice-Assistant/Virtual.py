import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import pyaudio
import os
import sys

# === Initialize recognizer and TTS engine ===
listener = sr.Recognizer()
machine = pyttsx3.init()                        

# === Set speaking rate and voice ===
machine.setProperty('rate', 150)
voices = machine.getProperty('voices')
machine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)

# === Speak Text ===
def talk(text):
    print(f"Assistant: {text}")
    machine.say(text)
    machine.runAndWait()

# === Listen for Voice Input ===
def input_instruction():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            listener.adjust_for_ambient_noise(source, duration=1)
            audio = listener.listen(source, timeout=5, phrase_time_limit=5)
            instruction = listener.recognize_google(audio).lower()
            print("🗣️ You said:", instruction)
            return instruction
    except sr.WaitTimeoutError:
        talk("Timeout. Please try again.")
    except sr.UnknownValueError:
        talk("Sorry, I didn't catch that.")
    except sr.RequestError:
        talk("Network error.")
    except Exception as e:
        talk(f"An error occurred: {str(e)}")
    return ""

# === Open Application ===
def open_app(path, app_name):
    if os.path.exists(path):
        os.startfile(path)
        talk(f"Opening {app_name}")
    else:
        talk(f"{app_name} not found on your system.")

# === Main Assistant Function ===
def play_bixby():
    talk("Hello, I am Bixby! What can I do for you?")
    
    while True:
        instruction = input_instruction()

        if not instruction:
            continue

        # === Exit Conditions ===
        if any(word in instruction for word in ['bye', 'exit', 'quit', 'close']):
            talk("Goodbye! Have a great day.")
            break

        # === Commands ===
        elif "play" in instruction:
            song = instruction.replace("play", "").strip()
            if song:
                talk("Playing " + song)
                pywhatkit.playonyt(song)
            else:
                talk("Please say the name of the song.")

        elif "time" in instruction:
            now = datetime.datetime.now().strftime('%I:%M %p')
            talk("The time is " + now)

        elif "date" in instruction:
            today = datetime.datetime.now().strftime('%B %d, %Y')
            talk("Today's date is " + today)

        elif "how are you" in instruction:
            talk("I'm doing great! How about you?")

        elif "your name" in instruction or "what is your name" in instruction:
            talk("My name is Bixby, your virtual assistant.")

        elif "who is" in instruction:
            person = instruction.replace("who is", "").strip()
            if person:
                try:
                    info = wikipedia.summary(person, sentences=1)
                    talk(info)
                except:
                    talk("Sorry, I couldn't find information.")
            else:
                talk("Please mention the name after 'who is'.")

        elif "joke" in instruction:
            joke = pyjokes.get_joke()
            talk(joke)

        # === App Opening Commands ===
        elif "open microsoft edge" in instruction:
            open_app("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "Microsoft Edge")

        elif "open chrome" in instruction or "open google" in instruction:
            talk("Opening Chrome")
            os.system("start chrome")

        elif "open brave" in instruction:
            open_app("C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe", "Brave Browser")

        elif "open visual studio code" in instruction or "open vs code" in instruction:
            open_app("C:\\Users\\Hp\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe", "Visual Studio Code")

        elif "open whatsapp" in instruction:
            talk("Opening WhatsApp")
            os.system("start whatsapp")

        elif "open youtube" in instruction:
            pywhatkit.playonyt("YouTube")

        else:
            talk("I didn't understand that. Can you repeat?")

# === Entry Point ===
if __name__ == "__main__":
    try:
        play_bixby()
    except Exception as e:
        print("❌ Fatal Error:", e)
        talk("Something went wrong. Please try again.")
        sys.exit(1)

