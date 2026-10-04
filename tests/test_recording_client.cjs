// node tests/test_recording_client.cjs
const {readFileSync} = require('node:fs');
const {runInNewContext} = require('node:vm');
const assert = require('node:assert/strict');
const sent = [];
runInNewContext(readFileSync('server/static/recording-client.js', 'utf8') + `
const recorder = new TrainingRecorder({socket: () => ({readyState: 1, send: x => sent.push(JSON.parse(x))}), key: 'test'});
recorder.active = true;
recorder.recordingId = 'test-recording';
recorder.handle({type: 'clock_sync', client_event_id: 'original', server_received_ns: 123, server_sent_ns: 456});
`, {sent, localStorage: {getItem: () => null}, crypto: require('node:crypto').webcrypto,
performance, WebSocket: {OPEN: 1}});
assert.equal(sent.length, 1);
assert.equal(sent[0].type, 'client_event');
assert.equal(sent[0].event, 'clock_reply');
assert.equal(sent[0].server_received_ns, 123);
assert.equal(sent[0].recording_id, 'test-recording');
console.log('Clock acknowledgement stays a client observation: passed');
