# Information About .rec Files

- 25 files, each corresponding to one patient.
- On average, each individual was analyzed for 6.93 hours.
- All patients had the following signals analyzed (some had others analyzed as well):
  - 'C3A2', 'abdo', 'Flow', 'Sum', 'Pulse', 'RightEye', 'C4A1', 'SpO2', 'BodyPos', 'ribcage', 'EMG', 'ECG', and 'Lefteye'.
- I have no idea of what some fo these represent, for example the body position and rib cage movements ones.
- Some interesting ones are ECG (electro cardiogram), EMG (electromyography), RightEye, and LeftEye (Electrooculography EOG)
- Each signal was analyzed approximately every 7.8 ms.
- On average, each type of signal was recorded 3195381.76 times for a patient (this is directly related to the amount of time a patient was studied).
- After looking at the Pulse data for one patient, I noticed that at the beginning and end of the time series theres a lot of observations close to 0 (or lower even). This somewhat makes sense and probably represents the period of time were the device was being set for the patient.
