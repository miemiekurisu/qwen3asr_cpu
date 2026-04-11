(function (global) {
  "use strict";

  function readFourCc(view, offset) {
    return String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
  }

  function clampToInt16(value) {
    if (value > 32767) {
      return 32767;
    }
    if (value < -32768) {
      return -32768;
    }
    return value;
  }

  function floatToInt16(value) {
    const clamped = Math.max(-1, Math.min(1, value));
    return clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
  }

  function concatFloat32(left, right) {
    const output = new Float32Array(left.length + right.length);
    output.set(left, 0);
    output.set(right, left.length);
    return output;
  }

  function parseWavHeader(buffer) {
    const view = buffer instanceof DataView ? buffer : new DataView(buffer);
    if (view.byteLength < 44) {
      throw new Error("WAV 头过短");
    }
    if (readFourCc(view, 0) !== "RIFF" || readFourCc(view, 8) !== "WAVE") {
      throw new Error("仅支持 RIFF/WAVE 文件");
    }

    let offset = 12;
    let format = null;
    let dataOffset = -1;
    let dataSize = 0;

    while (offset + 8 <= view.byteLength) {
      const chunkId = readFourCc(view, offset);
      const chunkSize = view.getUint32(offset + 4, true);
      const chunkDataOffset = offset + 8;
      const nextOffset = chunkDataOffset + chunkSize + (chunkSize & 1);
      if (nextOffset > view.byteLength && chunkId !== "data") {
        break;
      }

      if (chunkId === "fmt ") {
        if (chunkSize < 16 || chunkDataOffset + 16 > view.byteLength) {
          throw new Error("WAV fmt chunk 不完整");
        }
        const audioFormat = view.getUint16(chunkDataOffset, true);
        const channels = view.getUint16(chunkDataOffset + 2, true);
        const sampleRate = view.getUint32(chunkDataOffset + 4, true);
        const byteRate = view.getUint32(chunkDataOffset + 8, true);
        const blockAlign = view.getUint16(chunkDataOffset + 12, true);
        const bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
        format = {
          audioFormat,
          channels,
          sampleRate,
          byteRate,
          blockAlign,
          bitsPerSample,
        };
      } else if (chunkId === "data") {
        dataOffset = chunkDataOffset;
        dataSize = chunkSize;
        break;
      }

      offset = nextOffset;
    }

    if (!format) {
      throw new Error("缺少 WAV fmt chunk");
    }
    if (dataOffset < 0) {
      throw new Error("缺少 WAV data chunk");
    }
    if (format.audioFormat !== 1) {
      throw new Error("当前仅支持 PCM WAV 流式上传");
    }
    if (format.bitsPerSample !== 16) {
      throw new Error("当前仅支持 16-bit PCM WAV 流式上传");
    }
    if (format.channels <= 0 || format.blockAlign <= 0) {
      throw new Error("WAV 声道或 blockAlign 非法");
    }

    const bytesPerFrame = format.blockAlign;
    const frameCount = Math.floor(dataSize / bytesPerFrame);
    return {
      audioFormat: format.audioFormat,
      channels: format.channels,
      sampleRate: format.sampleRate,
      byteRate: format.byteRate,
      blockAlign: format.blockAlign,
      bitsPerSample: format.bitsPerSample,
      dataOffset,
      dataSize,
      bytesPerFrame,
      frameCount,
      durationSeconds: frameCount / format.sampleRate,
    };
  }

  function createMonoPcm16Transformer(format, targetSampleRate) {
    return {
      format,
      targetSampleRate: targetSampleRate || 16000,
      resampleStep: format.sampleRate / (targetSampleRate || 16000),
      resampleBuffer: new Float32Array(0),
      resamplePosition: 0,
    };
  }

  function convertDirectChunkToMonoPcm16(buffer, format) {
    const view = new DataView(buffer);
    const frameCount = Math.floor(view.byteLength / format.bytesPerFrame);
    const output = new Int16Array(frameCount);
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frameBase = frameIndex * format.bytesPerFrame;
      let sum = 0;
      for (let channelIndex = 0; channelIndex < format.channels; channelIndex += 1) {
        sum += view.getInt16(frameBase + channelIndex * 2, true);
      }
      output[frameIndex] = clampToInt16(Math.round(sum / format.channels));
    }
    return output;
  }

  function convertChunkToMonoFloats(buffer, format) {
    const view = new DataView(buffer);
    const frameCount = Math.floor(view.byteLength / format.bytesPerFrame);
    const output = new Float32Array(frameCount);
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frameBase = frameIndex * format.bytesPerFrame;
      let sum = 0;
      for (let channelIndex = 0; channelIndex < format.channels; channelIndex += 1) {
        sum += view.getInt16(frameBase + channelIndex * 2, true);
      }
      output[frameIndex] = (sum / format.channels) / 32768;
    }
    return output;
  }

  function resampleChunkToMonoPcm16(buffer, transformer, finalize) {
    const monoChunk = convertChunkToMonoFloats(buffer, transformer.format);
    const combined = concatFloat32(transformer.resampleBuffer, monoChunk);
    const samples = [];
    while (
      transformer.resamplePosition + 1 < combined.length ||
      (finalize && transformer.resamplePosition < combined.length)
    ) {
      const leftIndex = Math.floor(transformer.resamplePosition);
      const frac = transformer.resamplePosition - leftIndex;
      const rightIndex = Math.min(leftIndex + 1, combined.length - 1);
      const sample = combined[leftIndex] +
        (combined[rightIndex] - combined[leftIndex]) * frac;
      samples.push(floatToInt16(sample));
      transformer.resamplePosition += transformer.resampleStep;
    }

    const consumed = Math.floor(transformer.resamplePosition);
    transformer.resampleBuffer = combined.slice(consumed);
    transformer.resamplePosition -= consumed;
    return Int16Array.from(samples);
  }

  function convertChunkToMonoPcm16(buffer, transformer, finalize) {
    if (transformer.format.sampleRate === transformer.targetSampleRate) {
      return convertDirectChunkToMonoPcm16(buffer, transformer.format);
    }
    return resampleChunkToMonoPcm16(buffer, transformer, finalize);
  }

  function getChunkByteRange(format, frameOffset, frameCount) {
    const byteOffset = format.dataOffset + frameOffset * format.bytesPerFrame;
    return {
      start: byteOffset,
      end: byteOffset + frameCount * format.bytesPerFrame,
    };
  }

  const api = {
    parseWavHeader,
    createMonoPcm16Transformer,
    convertChunkToMonoPcm16,
    getChunkByteRange,
  };

  global.QasrWavUpload = api;
})(typeof globalThis !== "undefined" ? globalThis : this);