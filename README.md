# SoundEventClassification
SEC for Sri lanka Universities data set + Environmental data

In this project, I aim to detect and classify the immediate sounds events for home tools. 
Moreover, I am planning to understand the immediate sound events in order to have short and certain response time. 
In addition to detecting sound events I will also derive the direction of the sound event. 
Our major setback is background noise which is a part of the cocktail party program for many years. 
To overcome this obscurity we will first define the event that we will detect and then we train our algorithm in order to detect predefined sound classes.,

By using maixduino’s Esp 32 wifi module I opened up an audio stream port using Maixduino’s Standalone SDK in C++. 
Then using the pyaudio library we get the microphone audio stream and use the pretrained model in discrete audio clips to classify sound events.
We saved the microphone input stream buffering into a temporary buffer and worked on the buffer.
We set the chunk size and sample rate as 22050 hz to get the audio stream as a whole array and assumed that the first 1000 points belonged to the background noise
and reduced the noise in real time. We classified events and gave them a label along with the timestamp of the respected audio window in the stream.
We split  the audio stream into single sound events containing audio clips.
To split the stream correctly we write a function to determine silent parts in the clip which defines the start and end point of a clip.
Noting that inside sound events there could be silent parts we defined a threshold to prevent the small silence parts inside one sound event to count as another
sound event and printed the waveform of captured sound events. 


Feature Extraction
The preprocessed audio files themselves cannot be used to classify as sound events. 
We have to extract features from the audio clips. 
To do this we decided to use the absolute values of Short-Time Fourier Transform (STFT) from each audio clip to use the entire spectrum I decided not to work on the log mel band frequencies.
In STFT calculations, we choose 512 as the Fast Fourier transform window size(n_fft). 
According to signal theory n_stft = n_fft/2 + 1, 257 frequency bins(n_stft) are calculated over a window size of 512. 
The hop length of the window is chosen as 256 to have a better overlapping in calculating the STFT.

Part of my project for ELEC440 Advanced Sensors feel free to use the code and contact me 
Support Open Source!
ekin_boke@arcelik.com
