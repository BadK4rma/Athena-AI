import openai
import streamlit as st
import speech_recognition as sr
import spacy
import datetime
import pytz
import requests
import json
import os

# Set up OpenAI API credentials
openai.api_key = "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"

class Athena:
    def __init__(self):
        self.qa_pairs = {
            "name": "My name is Athena.",
            "weather": self.get_weather,
            "time": self.get_time,
            "reminder": self.set_reminder,
            "appointment": self.schedule_appointment,
            "email": self.send_email
        }
        self.nlp = spacy.load("en_core_web_sm")
        self.r = sr.Recognizer()
        self.q_values = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.load_q_values()

    def handle_query(self, text):
        if not text:
            return "I'm sorry, I didn't catch that. Could you please repeat?"

        text = text.lower()
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return f"Hello, {ent.text}!"

        for key in self.qa_pairs:
            if key in text:
                if callable(self.qa_pairs[key]):
                    return self.qa_pairs[key]()
                else:
                    return self.qa_pairs[key]

        if text not in self.q_values:
            self.q_values[text] = 0

        max_q = max(self.q_values.values())

        if max_q == 0:
            prompt = f"Can you please help me with {text}?"
            response = self.generate_response(prompt)
            self.q_values[text] += self.alpha * (0 - self.q_values[text])
            return response
        else:
            options = [k for k, v in self.q_values.items() if v == max_q]
            option = options[0]
            response = self.generate_response(option)
            self.q_values[text] += self.alpha * (self.gamma * self.q_values[option] - self.q_values[text])
            return response

    def get_time(self):
        timezone = pytz.timezone('US/Pacific')
        current_time = datetime.datetime.now(timezone).strftime("%I:%M %p")
        return f"The current time is {current_time}."

    def get_weather(self):
        api_key = os.getenv("WEATHER_API_KEY")
        url = f'http://api.openweathermap.org/data/2.5/weather?q=Los Angeles&appid={api_key}'

        try:
            response = requests.get(url)

            if response.status_code != 200:
                return "Sorry, I could not fetch the weather information at the moment."

            weather_data = response.json()
            description = weather_data['weather'][0]['description']
            temperature = round((weather_data['main']['temp'] - 273.15) * 9 / 5 + 32, 2)
            return f"The weather today is {description}, with a temperature of {temperature}°F."
        except:
            return "Sorry, I could not fetch the weather information at the moment."

    def set_reminder(self):
        return "Sorry, setting reminders is not yet supported."

    def schedule_appointment(self):
        return "Sorry, scheduling appointments is not yet supported."
    def send_email(self):
        return "Sorry, sending emails is not yet supported."

    def load_q_values(self):
        if os.path.exists("q_values.json"):
            with open("q_values.json", "r") as f:
                self.q_values = json.load(f)

    def save_q_values(self):
        with open("q_values.json", "w") as f:
            json.dump(self.q_values, f)

    def generate_response(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()

def main():
    

    st.title("Athena")
    st.write("Ask me anything! You can type your question or ask me verbally by clicking the button below.")

    athena = Athena()

    if False:
        with sr.Microphone() as source:
            audio = athena.r.listen(source)
        try:
            text = athena.r.recognize_google(audio)
            st.write(f"You said: {text}")
            response = athena.handle_query(text)
            st.write(f"Athena: {response}")
        except sr.UnknownValueError:
            st.write("I'm sorry, I didn't catch that. Could you please repeat?")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

    else:
        text = st.text_input("Type your question here")
        if text:
            response = athena.handle_query(text)
            st.write(f"Athena: {response}")

    athena.save_q_values()

if __name__ == "__main__":
    main()
