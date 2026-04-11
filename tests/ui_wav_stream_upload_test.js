ObjC.import("Foundation");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function assertNear(actual, expected, tolerance, message) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function expectThrows(fn, expectedMessage) {
  let thrown = null;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  assert(thrown, `expected throw: ${expectedMessage}`);
  assert(
    String(thrown.message).indexOf(expectedMessage) >= 0,
    `throw mismatch: expected message containing '${expectedMessage}', got '${thrown.message}'`,
  );
}

function loadText(path) {
  return $.NSString.stringWithContentsOfFileEncodingError($(path), $.NSUTF8StringEncoding, null).js;
}

function loadApi(modulePath) {
  (0, eval)(loadText(modulePath));
  return QasrWavUpload;
}

function concatUint8Arrays(parts) {
  let total = 0;
  for (let index = 0; index < parts.length; index += 1) {
    total += parts[index].length;
  }
  const output = new Uint8Array(total);
  let offset = 0;
  for (let index = 0; index < parts.length; index += 1) {
    output.set(parts[index], offset);
    offset += parts[index].length;
  }
  return output;
}

function encodeAscii(text) {
  const output = new Uint8Array(text.length);
  for (let index = 0; index < text.length; index += 1) {
    output[index] = text.charCodeAt(index);
  }
  return output;
}

function encodeInt16LE(values) {
  const output = new Uint8Array(values.length * 2);
  const view = new DataView(output.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setInt16(index * 2, values[index], true);
  }
  return output;
}

function encodeChunk(id, payload) {
  const paddedPayload = payload.length % 2 === 0
    ? payload
    : concatUint8Arrays([payload, new Uint8Array([0])]);
  const output = new Uint8Array(8 + paddedPayload.length);
  output.set(encodeAscii(id), 0);
  const view = new DataView(output.buffer);
  view.setUint32(4, payload.length, true);
  output.set(paddedPayload, 8);
  return output;
}

function buildPcmWav(options) {
  const channels = options.channels;
  const sampleRate = options.sampleRate;
  const samples = options.samples;
  const bitsPerSample = 16;
  const blockAlign = channels * (bitsPerSample / 8);
  const byteRate = sampleRate * blockAlign;
  const fmtPayload = new Uint8Array(16);
  const fmtView = new DataView(fmtPayload.buffer);
  fmtView.setUint16(0, 1, true);
  fmtView.setUint16(2, channels, true);
  fmtView.setUint32(4, sampleRate, true);
  fmtView.setUint32(8, byteRate, true);
  fmtView.setUint16(12, blockAlign, true);
  fmtView.setUint16(14, bitsPerSample, true);

  const chunks = [encodeChunk("fmt ", fmtPayload)];
  if (options.extraChunk) {
    chunks.push(encodeChunk(options.extraChunk.id, options.extraChunk.payload));
  }
  chunks.push(encodeChunk("data", encodeInt16LE(samples)));

  const chunkBytes = concatUint8Arrays(chunks);
  const header = new Uint8Array(12);
  header.set(encodeAscii("RIFF"), 0);
  new DataView(header.buffer).setUint32(4, 4 + chunkBytes.length, true);
  header.set(encodeAscii("WAVE"), 8);

  return concatUint8Arrays([header, chunkBytes]).buffer;
}

function manualMonoAverage(interleaved, channels) {
  const frames = interleaved.length / channels;
  const output = new Int16Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      sum += interleaved[frame * channels + channel];
    }
    output[frame] = Math.round(sum / channels);
  }
  return output;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function run(argv) {
  const api = loadApi(argv[0]);

  const normalWav = buildPcmWav({
    channels: 1,
    sampleRate: 16000,
    samples: [1000, -1000, 500, -500],
  });
  const header = api.parseWavHeader(normalWav);
  assertEqual(header.channels, 1, "normal header channels");
  assertEqual(header.sampleRate, 16000, "normal header sample rate");
  assertEqual(header.frameCount, 4, "normal header frame count");
  assertNear(header.durationSeconds, 4 / 16000, 1e-9, "normal header duration");

  const extremeWav = buildPcmWav({
    channels: 2,
    sampleRate: 48000,
    samples: [],
    extraChunk: {id: "JUNK", payload: new Uint8Array([1, 2, 3])},
  });
  const extremeHeader = api.parseWavHeader(extremeWav);
  assertEqual(extremeHeader.channels, 2, "extreme header channels");
  assertEqual(extremeHeader.frameCount, 0, "extreme header zero frames");

  expectThrows(() => api.parseWavHeader(new Uint8Array(32).buffer), "WAV 头过短");
  expectThrows(() => {
    const broken = new Uint8Array(normalWav);
    broken[0] = 0;
    api.parseWavHeader(broken.buffer);
  }, "仅支持 RIFF/WAVE 文件");

  const stereoSamples = [1000, -1000, 3000, -3000, 2000, 1000, -3000, -1000];
  const stereoWav = buildPcmWav({
    channels: 2,
    sampleRate: 16000,
    samples: stereoSamples,
  });
  const stereoHeader = api.parseWavHeader(stereoWav);
  const stereoTransformer = api.createMonoPcm16Transformer(stereoHeader, 16000);
  const stereoRange = api.getChunkByteRange(stereoHeader, 0, stereoHeader.frameCount);
  const stereoChunk = stereoWav.slice(stereoRange.start, stereoRange.end);
  const directMono = api.convertChunkToMonoPcm16(stereoChunk, stereoTransformer, true);
  const expectedMono = manualMonoAverage(stereoSamples, 2);
  assertEqual(directMono.length, expectedMono.length, "direct mono length");
  for (let index = 0; index < directMono.length; index += 1) {
    assertEqual(directMono[index], expectedMono[index], `direct mono sample ${index}`);
  }

  const random = seededRandom(7);
  const randomStereo = [];
  for (let index = 0; index < 256; index += 1) {
    randomStereo.push(Math.floor((random() * 65535) - 32768));
    randomStereo.push(Math.floor((random() * 65535) - 32768));
  }
  const randomWav = buildPcmWav({
    channels: 2,
    sampleRate: 16000,
    samples: randomStereo,
  });
  const randomHeader = api.parseWavHeader(randomWav);
  const randomTransformer = api.createMonoPcm16Transformer(randomHeader, 16000);
  const randomRange = api.getChunkByteRange(randomHeader, 0, randomHeader.frameCount);
  const randomMono = api.convertChunkToMonoPcm16(
    randomWav.slice(randomRange.start, randomRange.end),
    randomTransformer,
    true,
  );
  const randomExpected = manualMonoAverage(randomStereo, 2);
  for (let index = 0; index < randomMono.length; index += 1) {
    assertEqual(randomMono[index], randomExpected[index], `random mono sample ${index}`);
  }

  const resampleSamples = [0, 32767, 0, -32768];
  const resampleWav = buildPcmWav({
    channels: 1,
    sampleRate: 8000,
    samples: resampleSamples,
  });
  const resampleHeader = api.parseWavHeader(resampleWav);
  const oneShotTransformer = api.createMonoPcm16Transformer(resampleHeader, 16000);
  const resampleRange = api.getChunkByteRange(resampleHeader, 0, resampleHeader.frameCount);
  const oneShot = api.convertChunkToMonoPcm16(
    resampleWav.slice(resampleRange.start, resampleRange.end),
    oneShotTransformer,
    true,
  );
  assert(oneShot.length >= 7, "resample one-shot length");

  const splitTransformer = api.createMonoPcm16Transformer(resampleHeader, 16000);
  const firstHalfRange = api.getChunkByteRange(resampleHeader, 0, 2);
  const secondHalfRange = api.getChunkByteRange(resampleHeader, 2, 2);
  const splitFirst = api.convertChunkToMonoPcm16(
    resampleWav.slice(firstHalfRange.start, firstHalfRange.end),
    splitTransformer,
    false,
  );
  const splitSecond = api.convertChunkToMonoPcm16(
    resampleWav.slice(secondHalfRange.start, secondHalfRange.end),
    splitTransformer,
    true,
  );
  const splitCombined = new Int16Array(splitFirst.length + splitSecond.length);
  splitCombined.set(splitFirst, 0);
  splitCombined.set(splitSecond, splitFirst.length);
  assertEqual(splitCombined.length, oneShot.length, "split resample length");
  for (let index = 0; index < oneShot.length; index += 1) {
    assertEqual(splitCombined[index], oneShot[index], `split resample sample ${index}`);
  }

  console.log("ui_wav_stream_upload_test: PASS");
}