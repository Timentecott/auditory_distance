# Manual Presentation Level Control
# Play sounds through headphones and loudspeakers consecutively
# Ask which was louder and adjust levels accordingly

from psychopy import visual, event, core
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Audio file to use (noise)
BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
NOISE_FILE = os.path.join(BASE_DIR, 'original_audios', 'noise', 'brown_noise_5s.wav')

# Audio device indices (adjust these to match your setup)
HEADPHONES_DEVICE = 4
SPEAKERS_DEVICE = 5

# Initial gain levels (in dB)
headphone_gain_db = 0.0
speaker_gain_db = 0.0

# Gain adjustment step (in dB)
GAIN_STEP = 1.0

# ============================================================================
# SETUP
# ============================================================================

# Create PsychoPy window
win = visual.Window(
    size=(1024, 768),
    units='pix',
    fullscr=True,
    color=(0, 0, 0),
    allowStencil=False
)
win.mouseVisible = False

# Load the noise file
print(f"Loading audio file: {NOISE_FILE}")
try:
    audio_data, fs = sf.read(NOISE_FILE)
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"Audio loaded: {len(audio_data)/fs:.2f} seconds, {fs} Hz")
except Exception as e:
    print(f"Error loading audio file: {e}")
    win.close()
    core.quit()

# Create visual stimuli
instructions = visual.TextStim(
    win,
    text="You will hear two sounds.\n\nPress LEFT ARROW if FIRST was louder\nPress RIGHT ARROW if SECOND was louder\nPress R to repeat\nPress ESCAPE to quit\n\nPress any key to start",
    color='white',
    height=30,
    wrapWidth=900
)

fixation = visual.TextStim(win, text='+', color='white', height=50)

prompt = visual.TextStim(
    win,
    text="Which was louder?\n\nLEFT = First    RIGHT = Second    R = Repeat",
    color='white',
    height=30
)

up_arrow = visual.TextStim(win, text='?\n\nSpeaker level increased', color='green', height=80)
down_arrow = visual.TextStim(win, text='?\n\nSpeaker level decreased', color='red', height=80)

status_text = visual.TextStim(
    win,
    text='',
    color='white',
    height=25,
    pos=(0, -200)
)

# ============================================================================
# FUNCTIONS
# ============================================================================

def apply_gain(audio, gain_db):
    """Apply gain in dB to audio signal"""
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear

def play_sound(audio, device, device_name):
    """Play audio through specified device"""
    print(f"  Playing through {device_name} (device {device})")
    sd.default.device = (None, device)
    sd.play(audio, samplerate=fs)
    sd.wait()

def play_comparison():
    """Play headphones then speakers and return"""
    # Apply current gains
    headphone_audio = apply_gain(audio_data, headphone_gain_db)
    speaker_audio = apply_gain(audio_data, speaker_gain_db)
    
    # Show fixation cross
    fixation.draw()
    win.flip()
    core.wait(0.5)
    
    # Play first (headphones)
    print("\nPlaying comparison:")
    print(f"  Headphone gain: {headphone_gain_db:+.1f} dB")
    play_sound(headphone_audio, HEADPHONES_DEVICE, "headphones")
    
    # Brief pause
    core.wait(0.5)
    
    # Play second (speakers)
    print(f"  Speaker gain: {speaker_gain_db:+.1f} dB")
    play_sound(speaker_audio, SPEAKERS_DEVICE, "speakers")
    
    # Brief pause before prompt
    core.wait(0.3)

def show_feedback(response):
    """Show feedback arrow based on response"""
    global speaker_gain_db
    
    if response == 'left':
        # First (headphones) was louder, decrease speaker level
        speaker_gain_db -= GAIN_STEP
        down_arrow.draw()
        status_text.setText(f"Speaker: {speaker_gain_db:+.1f} dB | Headphone: {headphone_gain_db:+.1f} dB")
        status_text.draw()
        win.flip()
        print(f"  ? Speaker level decreased to {speaker_gain_db:+.1f} dB")
    elif response == 'right':
        # Second (speakers) was louder, increase speaker level
        speaker_gain_db += GAIN_STEP
        up_arrow.draw()
        status_text.setText(f"Speaker: {speaker_gain_db:+.1f} dB | Headphone: {headphone_gain_db:+.1f} dB")
        status_text.draw()
        win.flip()
        print(f"  ? Speaker level increased to {speaker_gain_db:+.1f} dB")
    
    core.wait(1.5)

# ============================================================================
# MAIN LOOP
# ============================================================================

# Show instructions
instructions.draw()
win.flip()
event.waitKeys()

print("\n" + "="*70)
print("MANUAL LEVEL CONTROL SESSION")
print("="*70)
print(f"Headphones: device {HEADPHONES_DEVICE}")
print(f"Speakers: device {SPEAKERS_DEVICE}")
print(f"Initial gains: Headphones {headphone_gain_db:+.1f} dB, Speakers {speaker_gain_db:+.1f} dB")
print("="*70)

trial_count = 0

# Main loop
while True:
    trial_count += 1
    print(f"\n--- Trial {trial_count} ---")
    
    # Play the comparison
    play_comparison()
    
    # Show prompt and wait for response
    prompt.draw()
    status_text.setText(f"Speaker: {speaker_gain_db:+.1f} dB | Headphone: {headphone_gain_db:+.1f} dB")
    status_text.draw()
    win.flip()
    
    # Wait for response
    keys = event.waitKeys(keyList=['left', 'right', 'r', 'escape'])
    
    if 'escape' in keys:
        print("\nSession ended by user")
        break
    elif 'r' in keys:
        print("  Repeating trial...")
        continue
    elif keys[0] in ['left', 'right']:
        show_feedback(keys[0])

# ============================================================================
# CLEANUP
# ============================================================================

print("\n" + "="*70)
print("SESSION SUMMARY")
print("="*70)
print(f"Total trials: {trial_count}")
print(f"Final speaker gain: {speaker_gain_db:+.1f} dB")
print(f"Final headphone gain: {headphone_gain_db:+.1f} dB")
print(f"Difference: {speaker_gain_db - headphone_gain_db:+.1f} dB")
print("="*70)

# Close window
win.close()
core.quit()

