import csv
import os
import time
from pathlib import Path

os.environ["SD_ENABLE_ASIO"] = "1"  # this line is important as it allows revelation of asio devices

import soundfile as sf
import sounddevice as sd
import numpy as np
from psychopy import visual, event, core
# from conditions file
SCRIPT_DIR = Path(__file__).resolve().parent
conditions_file = SCRIPT_DIR / "conditions.csv"


def load_trial_conditions(conditions_path):
	with conditions_path.open("r", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		rows = list(reader)

	if not rows:
		raise ValueError(f"No condition rows found in {conditions_path}")

	return rows


def resolve_audio_path(playback_condition_value, stimulus_condition_value, file_name):
	playback_folder = playback_condition_value.strip()
	stimulus_folder = stimulus_condition_value.strip().lower()
	return SCRIPT_DIR  / playback_folder / stimulus_folder / f"{file_name}.wav"


trial_conditions = load_trial_conditions(conditions_file)

# set up 
playback_device = 20  # always use this device index
headphone_channels = (0, 1)  # output channels for headphones
loudspeaker_channels = (2, 3)  # output channels for loudspeakers
target_sr = 44100 # if this does not match, the script should fail
results_dir = SCRIPT_DIR / "results"
results_file = None
RESULTS_FIELDS = [
	"participant_id",
	"trial_number",
	"presentation_type",
	"stimulus_condition",
	"stimulus_file",
	"response",
	"accuracy",
	"rt",
	"timestamp",
]
stream_state = {
	"audio": None,
	"position": 0,
	"playback_mode": None,
	"target_pair": None,
}


def initialize_results_file(participant_number):
	global results_file
	results_dir.mkdir(exist_ok=True)
	results_file = results_dir / f"participant_{participant_number}_results.csv"
	if not results_file.exists():
		with results_file.open("w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=RESULTS_FIELDS)
			writer.writeheader()
	return results_file


def append_trial_result(participant_number, trial_number, presentation_type, stimulus_condition_value, stimulus_file_value, response, accuracy, rt):
	if results_file is None:
		raise RuntimeError("Results file has not been initialized.")

	with results_file.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=RESULTS_FIELDS)
		writer.writerow({
			"participant_id": participant_number,
			"trial_number": trial_number,
			"presentation_type": presentation_type,
			"stimulus_condition": stimulus_condition_value,
			"stimulus_file": stimulus_file_value,
			"response": response,
			"accuracy": accuracy,
			"rt": f"{rt:.3f}",
			"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
		})


def load_and_prepare(audio_file):
	audio_path = Path(audio_file)
	if not audio_path.is_file():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")

	data, sr = sf.read(str(audio_path), dtype='float32')
	if data.ndim == 1:
		data = data[:, np.newaxis]
	return data.astype(np.float32), sr

def validate_output_settings(device_index, sample_rate):
	try:
		sd.check_output_settings(device=device_index, samplerate=sample_rate, channels=4, dtype='float32')
	except Exception as exc:
		raise RuntimeError(
			f"Output settings are not supported for device {device_index} at {sample_rate} Hz: {exc}"
		)


def make_audio_stream(device_index, sample_rate):
	def callback(outdata, frames, time_info, status):
		if status:
			print(f"Audio stream status: {status}")

		audio = stream_state["audio"]
		position = stream_state["position"]
		playback_mode = stream_state["playback_mode"]
		target_pair = stream_state["target_pair"]

		outdata[:] = 0
		if audio is None or playback_mode is None or target_pair is None:
			return
		if position >= audio.shape[0]:
			return

		end = min(position + frames, audio.shape[0])
		chunk = audio[position:end]
		count = chunk.shape[0]
		if count > 0:
			if audio.ndim == 1 or audio.shape[1] == 1:
				left = chunk[:, 0] if chunk.ndim > 1 else chunk
				if playback_mode == "headphone":
					outdata[:count, target_pair[0]] = left
					outdata[:count, target_pair[1]] = left
				elif playback_mode == "loudspeaker":
					outdata[:count, target_pair[0]] = left
				else:
					raise ValueError(f"Unknown playback mode: {playback_mode}")
			else:
				if playback_mode == "headphone":
					outdata[:count, target_pair[0]] = chunk[:, 0]
					outdata[:count, target_pair[1]] = chunk[:, 1]
				elif playback_mode == "loudspeaker":
					outdata[:count, target_pair[0]] = chunk[:, 0]
				else:
					raise ValueError(f"Unknown playback mode: {playback_mode}")
		stream_state["position"] = end

	return sd.OutputStream(
		device=device_index,
		channels=4,
		samplerate=sample_rate,
		dtype='float32',
		callback=callback,
		latency='high',
		blocksize=8192,
	)


def get_accuracy_for_response(playback_mode, response):
	if playback_mode == "loudspeaker":
		return 1 if response == "up" else 0
	return 1 if response == "down" else 0


def run_trial(participant_number, trial_number, playback_condition, stimulus_condition_value, audio_path):
	validate_output_settings(playback_device, target_sr)

	audio_data, sr = load_and_prepare(audio_path)
	if sr != target_sr:
		raise ValueError(f"Audio sample rate {sr} Hz does not match required output sample rate {target_sr} Hz")

	if playback_condition in ("in_situ", "ex_situ"):
		playback_mode = "headphone"
		target_pair = headphone_channels
	elif playback_condition == "loudspeaker":
		playback_mode = "loudspeaker"
		target_pair = loudspeaker_channels
	else:
		raise ValueError(f"Unknown presentation condition: {playback_condition}")

	fixation = visual.TextStim(win, text="+", color='white', height=80)
	response_image = visual.ImageStim(win, image=str(SCRIPT_DIR / "resources" / "headphonevsloudspeak_info_graphic.png"))

	fixation.draw()
	win.flip()
	core.wait(2.0)

	stream_state["audio"] = audio_data
	stream_state["position"] = 0
	stream_state["playback_mode"] = playback_mode
	stream_state["target_pair"] = target_pair

	response = None
	response_clock = core.Clock()
	event.clearEvents()

	core.wait(3.0)
	response_image.draw()
	win.flip()
	response_clock.reset()
	event.clearEvents()

	while response is None:
		keys = event.getKeys()
		for key in keys:
			if key == 'escape':
				win.close()
				core.quit()
			elif key == 'up':
				response = 'up'
			elif key == 'down':
				response = 'down'
		if response is None:
			core.wait(0.01)

	rt = response_clock.getTime()
	accuracy = get_accuracy_for_response(playback_mode, response)
	append_trial_result(participant_number, trial_number, playback_condition, stimulus_condition_value, audio_path.name, response, accuracy, rt)
	return response, accuracy, rt


# launch psychopy window
win = visual.Window(
	size=(1024, 768),
	units='pix',
	fullscr=True,
	color=(0, 0, 0),
	allowStencil=False
)

# hide mouse cursor
win.mouseVisible = False


# ask for participant ID (keyboard input)
participant_id_prompt = visual.TextStim(win, text="Please enter your participant ID and press ENTER:", color='white')
participant_id_prompt.draw()
win.flip()

# Get participant ID input
participant_id = ""
keys = []
while True:
	keys = event.getKeys()
	for key in keys:
		if key == 'return':  # Enter key pressed
			if len(participant_id) > 0:
				break
		elif key == 'backspace':
			participant_id = participant_id[:-1]
		elif key == 'escape':
			win.close()
			core.quit()
		elif len(key) == 1 and key.isalnum():  # alphanumeric characters
			participant_id += key

		# Update display with current input
		input_text = f"Please enter your participant ID and press ENTER:\n\n{participant_id}"
		participant_id_prompt.setText(input_text)
		participant_id_prompt.draw()
		win.flip()

	if keys and 'return' in keys and len(participant_id) > 0:
		break

print(f"Participant ID: {participant_id}")  # Debug print
initialize_results_file(participant_id)
print(f"Results file created at: {results_file}")


if __name__ == "__main__":
	with make_audio_stream(playback_device, target_sr):
		for trial_number, row in enumerate(trial_conditions, start=1):
			playback_condition = row["Presentation_condition"].strip()
			stimulus_condition_value = row["Stimulus_condition"].strip()
			file_name = row["file"].strip()
			audio_path = resolve_audio_path(playback_condition, stimulus_condition_value, file_name)
			run_trial(participant_id, trial_number, playback_condition, stimulus_condition_value, audio_path)

