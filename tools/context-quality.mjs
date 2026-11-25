// Build exact shared token prompts and score the final answer without another model.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const expected = {
  badge: 'JASPER-6046',
  site: 'Cedar Annex',
  coolant: 'Beryl Mix',
  contact: 'Relay-17',
};

const records = [
  [6, '\n\nPelican Desk access register. Current gate badge: JASPER-6046. Previous badge ONYX-2991 is revoked.\n\n'],
  [28, '\n\nIris dispatch manifest. Active maintenance ticket NT-4728 supersedes the closed ticket NT-4721.\n\n'],
  [53, '\n\nTicket NT-4728: maintenance site Cedar Annex; approved coolant Beryl Mix. Ticket NT-4721 was for Willow Annex and Quartz Mix.\n\n'],
  [79, '\n\nSite directory: Cedar Annex contact channel is Relay-17. Willow Annex uses Relay-71.\n\n'],
];

function makeFixtures(binary, model, corpus, out) {
  mkdirSync(out, { recursive: true });
  const tokenize = (path, flags = []) => execFileSync(binary, ['tokenize', model, path, ...flags], {
    encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
  }).trim().split('\n');
  function part(name, text) {
    const path = join(out, name + '.txt');
    writeFileSync(path, text);
    return tokenize(path, ['--parse-special']);
  }
  const head = part('header', '<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Read this archive. Records for the questions are interspersed in the text.\n\n');
  const tail = part('question', '\n\nWhat is the current Pelican Desk gate badge? For the Iris dispatch’s active ticket, what is its maintenance site, approved coolant and contact channel for that site? Answer with one JSON object containing exactly the string keys badge, site, coolant and contact.<|end_message|><|message_model|>');
  const inserts = records.map(([percent, text]) => [percent, part('record-' + percent, text)]);
  const haystack = tokenize(corpus);
  for (const context of [32768, 65536, 131072]) {
    const count = context - 512;
    const ids = [...head];
    let used = 0;
    for (const [percent, record] of inserts) {
      const need = Math.floor(count * percent / 100) - ids.length;
      ids.push(...haystack.slice(used, used + need), ...record);
      used += need;
    }
    const remaining = count - ids.length - tail.length;
    if (haystack.length < used + remaining) throw new Error('Haystack corpus is too short');
    ids.push(...haystack.slice(used, used + remaining), ...tail);
    if (ids.length !== count) throw new Error('Incorrect prompt size');
    writeFileSync(join(out, context + '.ids'), ids.join('\n') + '\n');
  }
}

function score(path) {
  const output = readFileSync(path, 'utf8');
  const segments = output.split('<|content_text|>');
  const text = segments.length === 1 && output.includes('<|content_thinking|>') ? '' : segments.at(-1);
  let answer;
  for (const candidate of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      answer = JSON.parse(candidate[0]);
    } catch {
      // An incomplete or malformed object earns no credit.
    }
  }
  const hits = Object.entries(expected).map(([field, value]) =>
    typeof answer?.[field] === 'string' && answer[field].trim().toLowerCase() === value.toLowerCase());
  const facts = hits.slice(1).filter(Boolean).length;
  const complete = Object.keys(expected).every(field => typeof answer?.[field] === 'string');
  console.log([Number(hits[0]), facts, Number(facts === 3), Number(complete)].join('\t'));
}

const [mode, ...args] = process.argv.slice(2);
if (mode === 'make' && args.length === 4) makeFixtures(...args);
else if (mode === 'score' && args.length === 1) score(args[0]);
else {
  console.error('usage: node tools/context-quality.mjs make ABLA_GRAD MODEL CORPUS OUT_DIR | score RESPONSE');
  process.exitCode = 2;
}
