/**
 * Posner Cueing Task - PsychoJS Version
 * 
 * Experiment: Sound localization with valid/invalid spatial cues
 * - Participants hear a directional sound cue (4 locations)
 * - Then see a dot at one of 4 locations (valid=same as cue, invalid=opposite)
 * - Respond with arrow keys to indicate dot location
 * - Practice with feedback, main trials without feedback
 */

import { PsychoJS } from './lib/core-2023.2.0.js';
import * as core from './lib/core-2023.2.0.js';
import * as data from './lib/data-2023.2.0.js';
import * as sound from './lib/sound-2023.2.0.js';
import * as visual from './lib/visual-2023.2.0.js';
import * as util from './lib/util-2023.2.0.js';

// =============================================================================
// CONFIG
// =============================================================================

const NUMBER_OF_TRIALS = 64;
const FIXATION_DURATION = 0.5;
const SOUND_CUE_DURATION = 0.1;
const CUE_TO_DOT_ISI = 0.1;
const DOT_DURATION = 0.1;
const INTER_TRIAL_INTERVAL = 1.0;
const PRACTICE_TRIALS = 4;

const DISTANCE_FROM_CENTER = 200;
const LOCATIONS = {
  left: [-DISTANCE_FROM_CENTER, 0],
  right: [DISTANCE_FROM_CENTER, 0],
  up: [0, DISTANCE_FROM_CENTER],
  down: [0, -DISTANCE_FROM_CENTER]
};

const SOUND_PREFIX = 'audio_stimuli/Localised/pink_noise_48k_30s_300_8000hz';

// =============================================================================
// INITIALIZE PSYCHOJS
// =============================================================================

const psychoJS = new PsychoJS({
  debug: true
});

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function oppositeLocation(location) {
  const opposites = {
    left: 'right',
    right: 'left',
    up: 'down',
    down: 'up'
  };
  return opposites[location];
}

function shuffle(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function makePracticeTrials() {
  const locations = ['left', 'right', 'up', 'down'];
  const trials = [];
  for (let i = 0; i < locations.length; i++) {
    trials.push({
      sound_location: locations[i],
      is_valid: (i % 2 === 0),
      trial_number: i + 1
    });
  }
  return shuffle(trials);
}

function makeMainTrials(n_trials) {
  const locations = ['left', 'right', 'up', 'down'];
  const trialsPer = n_trials / 4;
  const validityPer = n_trials / 2;

  let soundLocs = [];
  for (let loc of locations) {
    for (let i = 0; i < trialsPer; i++) {
      soundLocs.push(loc);
    }
  }

  let validities = [];
  for (let i = 0; i < validityPer; i++) {
    validities.push(true);
  }
  for (let i = 0; i < validityPer; i++) {
    validities.push(false);
  }

  soundLocs = shuffle(soundLocs);
  validities = shuffle(validities);

  const trials = [];
  for (let i = 0; i < n_trials; i++) {
    const soundLoc = soundLocs[i];
    const isValid = validities[i];
    const dotLoc = isValid ? soundLoc : oppositeLocation(soundLoc);
    trials.push({
      trial_number: i + 1,
      sound_location: soundLoc,
      dot_location: dotLoc,
      is_valid: isValid,
      sound_file: `${SOUND_PREFIX}_${soundLoc}.wav`
    });
  }
  return trials;
}

// =============================================================================
// MAIN EXPERIMENT
// =============================================================================

async function runExperiment() {
  // Open window
  psychoJS.openWindow({
    fullscr: true,
    color: new util.Color('gray'),
    units: 'pix',
    waitBlanking: true
  });

  // Create visual elements
  const fixation = new visual.ShapeStim({
    win: psychoJS.window,
    name: 'fixation',
    vertices: 'cross',
    size: [30, 30],
    ori: 0.0,
    pos: [0, 0],
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: 1.0,
    depth: 0.0,
    interpolate: true
  });

  const dot = new visual.Circle({
    win: psychoJS.window,
    name: 'dot',
    radius: 20,
    ori: 0.0,
    pos: [0, 0],
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: 1.0,
    depth: 0.0,
    interpolate: true
  });

  const instructionsText = new visual.TextStim({
    win: psychoJS.window,
    name: 'instructions',
    text: 'You will hear a sound, then see a dot.\nPress arrow keys (LEFT, RIGHT, UP, DOWN) to indicate where the dot is as quickly as possible.\nUse one finger for the whole experiment.\n\nPress any key to begin.',
    font: 'Arial',
    units: 'pix',
    pos: [0, 0],
    height: 30.0,
    color: new util.Color('white'),
    colorSpace: 'rgb',
    opacity: 1.0,
    depth: 0.0,
    wrapWidth: 1000
  });

  const practiceInstructionsText = new visual.TextStim({
    win: psychoJS.window,
    name: 'practice_instructions',
    text: 'Practice trials will now begin.\nYou will get feedback on each trial.\nPress any key to start.',
    font: 'Arial',
    units: 'pix',
    pos: [0, 0],
    height: 30.0,
    color: new util.Color('white'),
    colorSpace: 'rgb',
    opacity: 1.0,
    depth: 0.0,
    wrapWidth: 1000
  });

  const mainInstructionsText = new visual.TextStim({
    win: psychoJS.window,
    name: 'main_instructions',
    text: 'Practice complete!\nMain experiment will now begin.\nNo feedback will be given.\nPress any key to start.',
    font: 'Arial',
    units: 'pix',
    pos: [0, 0],
    height: 30.0,
    color: new util.Color('white'),
    colorSpace: 'rgb',
    opacity: 1.0,
    depth: 0.0,
    wrapWidth: 1000
  });

  const feedbackText = new visual.TextStim({
    win: psychoJS.window,
    name: 'feedback',
    text: '',
    font: 'Arial',
    units: 'pix',
    pos: [0, 0],
    height: 40.0,
    color: new util.Color('white'),
    colorSpace: 'rgb',
    opacity: 1.0,
    depth: 0.0,
    wrapWidth: 1000
  });

  const endText = new visual.TextStim({
    win: psychoJS.window,
    name: 'end',
    text: 'Experiment complete!\n\nThank you for participating.',
    font: 'Arial',
    units: 'pix',
    pos: [0, 0],
    height: 30.0,
    color: new util.Color('white'),
    colorSpace: 'rgb',
    opacity: 1.0,
    depth: 0.0,
    wrapWidth: 1000
  });

  // Data logging
  const trialData = [];

  // Wait for start
  instructionsText.setAutoDraw(true);
  psychoJS.window.render();
  await new Promise((resolve) => {
    const handleKey = () => {
      document.removeEventListener('keydown', handleKey);
      resolve();
    };
    document.addEventListener('keydown', handleKey, { once: true });
  });
  instructionsText.setAutoDraw(false);

  // ===== PRACTICE TRIALS =====
  const practiceTrials = makePracticeTrials();

  practiceInstructionsText.setAutoDraw(true);
  psychoJS.window.render();
  await new Promise((resolve) => {
    const handleKey = () => {
      document.removeEventListener('keydown', handleKey);
      resolve();
    };
    document.addEventListener('keydown', handleKey, { once: true });
  });
  practiceInstructionsText.setAutoDraw(false);

  for (const trial of practiceTrials) {
    const result = await runTrial(
      trial,
      fixation,
      dot,
      feedbackText,
      true // showFeedback
    );
    trialData.push({ ...trial, ...result, trial_type: 'practice' });
  }

  // ===== MAIN TRIALS =====
  const mainTrials = makeMainTrials(NUMBER_OF_TRIALS);

  mainInstructionsText.setAutoDraw(true);
  psychoJS.window.render();
  await new Promise((resolve) => {
    const handleKey = () => {
      document.removeEventListener('keydown', handleKey);
      resolve();
    };
    document.addEventListener('keydown', handleKey, { once: true });
  });
  mainInstructionsText.setAutoDraw(false);

  for (const trial of mainTrials) {
    const result = await runTrial(
      trial,
      fixation,
      dot,
      feedbackText,
      false // no feedback
    );
    trialData.push({ ...trial, ...result, trial_type: 'main' });
  }

  // ===== END SCREEN =====
  endText.setAutoDraw(true);
  psychoJS.window.render();
  await core.wait(2.0);

  // Log data
  psychoJS.experiment.addData('trials', trialData);
  await psychoJS.experiment.save();

  // Cleanup
  psychoJS.quit({
    message: 'Experiment finished.',
    isCompleted: true
  });
}

// =============================================================================
// TRIAL RUNNER
// =============================================================================

async function runTrial(trial, fixation, dot, feedbackText, showFeedback) {
  const soundLocation = trial.sound_location;
  const dotLocation = trial.dot_location;
  const soundFile = trial.sound_file;

  // 1. Fixation
  fixation.setAutoDraw(true);
  psychoJS.window.render();
  await core.wait(FIXATION_DURATION);

  // 2. Sound cue
  const trialSound = new sound.Sound({
    win: psychoJS.window,
    value: soundFile,
    secs: SOUND_CUE_DURATION,
    stereo: true,
    hamming: true,
    preBuffer: 0.5
  });
  trialSound.play();
  await core.wait(SOUND_CUE_DURATION);
  trialSound.stop();

  // 3. ISI
  fixation.setAutoDraw(true);
  psychoJS.window.render();
  await core.wait(CUE_TO_DOT_ISI);

  // 4. Dot presentation
  dot.pos = LOCATIONS[dotLocation];
  dot.setAutoDraw(true);
  fixation.setAutoDraw(false);
  psychoJS.window.render();
  await core.wait(DOT_DURATION);

  // 5. Wait for response
  fixation.setAutoDraw(false);
  dot.setAutoDraw(false);
  psychoJS.window.render();

  const responseStart = core.getTime();
  let response = null;
  let responseTime = null;

  await new Promise((resolve) => {
    const handleKey = (event) => {
      const key = event.key.toLowerCase();
      const keyMap = {
        arrowleft: 'left',
        arrowright: 'right',
        arrowup: 'up',
        arrowdown: 'down'
      };

      if (key in keyMap) {
        response = keyMap[key];
        responseTime = core.getTime() - responseStart;
        document.removeEventListener('keydown', handleKey);
        resolve();
      }
    };
    document.addEventListener('keydown', handleKey);
  });

  // 6. Feedback (if practice)
  if (showFeedback && response) {
    const correct = response === dotLocation;
    feedbackText.setText(correct ? 'Correct!' : 'Incorrect');
    feedbackText.setColor(correct ? new util.Color('green') : new util.Color('red'));
    feedbackText.setAutoDraw(true);
    psychoJS.window.render();
    await core.wait(0.5);
    feedbackText.setAutoDraw(false);
  } else {
    await core.wait(0.5);
  }

  // 7. ITI
  psychoJS.window.render();
  await core.wait(INTER_TRIAL_INTERVAL);

  // Return trial result
  const correct = response === dotLocation;
  return {
    response: response,
    response_time: responseTime,
    correct: correct ? 1 : 0
  };
}

// =============================================================================
// START
// =============================================================================

runExperiment().catch(error => {
  console.error('Experiment error:', error);
  psychoJS.quit({ message: 'Error occurred', isCompleted: false });
});
