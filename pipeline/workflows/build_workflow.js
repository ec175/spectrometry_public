// Generates batch-shorts.json (the main pipeline workflow) as clean n8n JSON.
// Authoring in JS avoids hand-escaping embedded expressions/prompts.
//   node build_workflow.js   ->  writes batch-shorts.json next to this file
const fs = require('fs');
const path = require('path');

// Fill these in for your own n8n instance. The credential IDs come from your
// n8n Credentials (each credential's id), and RENDER_TOKEN must match the token
// in render_service/token.txt. None of these are committed with real values.
const ANTHROPIC_CRED = { id: process.env.ANTHROPIC_CRED_ID || 'YOUR_ANTHROPIC_CRED_ID', name: 'Anthropic API' };
const YOUTUBE_CRED   = { id: process.env.YOUTUBE_CRED_ID  || 'YOUR_YOUTUBE_CRED_ID',  name: 'YouTube account' };
const RENDER_TOKEN   = process.env.RENDER_TOKEN || 'YOUR_RENDER_SERVICE_TOKEN';
const MODEL          = 'claude-sonnet-4-6';

const KNOWN = ['Caffeine','Tylenol','Benadryl','Dopamine','Adrenaline','Serotonin','GABA',
  'Glutamate','Glutamine','Glycine','Alanine','Valine','Leucine','Isoleucine','Proline',
  'Serine','Threonine','Cysteine','Methionine','Asparagine','Aspartate','Lysine','Arginine',
  'Histidine','Phenylalanine','Tyrosine','Tryptophan','Glucose','Sucrose','CitricAcid',
  'Vanillin','Menthol','Capsaicin','Ibuprofen','VitaminC','Morphine','Penicillin',
  'Cholesterol','Testosterone','Melatonin','Nicotine','CBD'].join(', ');

// ---- AI request bodies, built inside expressions via JSON.stringify ----------
const parseSystem =
  'You convert a request into manim scene class names. Known molecules (CamelCase): ' + KNOWN +
  '. Each has a per-molecule composite scene class named Stacked_<Name>. Given a free-text ' +
  'molecule list and a template, return ONLY a JSON array (no prose, no markdown fences), each ' +
  'element {"molecule":"<Name>","scene_class":"Stacked_<Name>"}. Normalize each requested ' +
  'molecule to the CLOSEST known molecule. The template does not change the class name (the ' +
  'Stacked_ composite already shows all methods).';

const metaSystem =
  'You write SEO metadata for short, educational chemistry videos: animated spectroscopy ' +
  'explainers (FTIR/IR, UV-Vis, Raman, NMR, PXRD) built with manim in Python. Given a molecule, ' +
  'return ONLY JSON (no markdown fences): {"title": a catchy <=90 char title that includes the ' +
  'molecule name, "description": ONE sentence, MAXIMUM 100 characters, packed with the top terms ' +
  'someone would SEARCH for this molecule (common + chemical name, what it is), "tags": array of ' +
  '8-15 plain keyword phrases WITHOUT the # symbol, "hashtags": array of hashtag words WITHOUT ' +
  'the # prefix}. Keep the description at or under 100 characters. The hashtags array MUST ' +
  'contain these exact words (plus a few extra relevant ones): manim, chemistry, python, ' +
  '<Molecule>, stem.';

// expression strings (start with '=') -----------------------------------------
const parseBodyExpr =
  "={{ JSON.stringify({ model: '" + MODEL + "', max_tokens: 700, system: " +
  JSON.stringify(parseSystem) +
  ", messages: [ { role: 'user', content: 'Molecules: ' + $json.Molecules + ' | Template: ' + $json.Template } ] }) }}";

const metaBodyExpr =
  "={{ JSON.stringify({ model: '" + MODEL + "', max_tokens: 800, system: " +
  JSON.stringify(metaSystem) +
  ", messages: [ { role: 'user', content: 'Molecule: ' + $json.molecule + '. Methods shown: FTIR, UV-Vis, Raman, NMR, PXRD.' } ] }) }}";

const parseListCode = [
  "// Anthropic returns { content: [ { type:'text', text } ] }. Fan out one item per molecule.",
  "const text = ($json.content && $json.content[0] && $json.content[0].text) || '[]';",
  "let arr;",
  "try { arr = JSON.parse(text); } catch (e) { throw new Error('AI parse did not return JSON: ' + text); }",
  "if (!Array.isArray(arr)) throw new Error('AI parse JSON was not an array');",
  "return arr.map(o => ({ json: { molecule: o.molecule, scene_class: o.scene_class }, pairedItem: { item: 0 } }));",
].join('\n');

const buildMetaCode = [
  "// Runs once per item. Combine the AI metadata with the carried molecule/scene_class.",
  "const text = ($json.content && $json.content[0] && $json.content[0].text) || '{}';",
  "let m; try { m = JSON.parse(text); } catch (e) { throw new Error('AI metadata not JSON: ' + text); }",
  "const molecule = $('Parse list').item.json.molecule;",
  "const scene_class = $('Parse list').item.json.scene_class;",
  "const required = ['manim','chemistry','python', String(molecule).replace(/[^A-Za-z0-9]/g,''), 'stem'];",
  "const tags = Array.isArray(m.hashtags) ? m.hashtags.slice() : [];",
  "const lower = tags.map(t => String(t).toLowerCase());",
  "for (const r of required) if (!lower.includes(r.toLowerCase())) tags.push(r);",
  "const hashtagLine = tags.map(t => '#' + String(t).replace(/^#/,'')).join(' ');",
  "// cap the AI description text to 100 chars, then append the hashtag line",
  "let desc = String(m.description || '').trim();",
  "if (desc.length > 100) desc = desc.slice(0, 100).replace(/\\s+\\S*$/, '').trim();",
  "const description = desc + '\\n\\n' + hashtagLine;",
  "const ytTags = (Array.isArray(m.tags) ? m.tags : []).map(t => String(t).replace(/^#/,'')).join(',');",
  "return { json: { molecule, scene_class, title: m.title, description, ytTags, hashtags: tags } };",
].join('\n');

// ---- nodes ------------------------------------------------------------------
const nodes = [
  {
    parameters: {
      formTitle: 'Batch shorts request',
      formDescription: 'List the molecules you want and pick a template. Missing videos are rendered automatically, then captioned and uploaded.',
      formFields: { values: [
        { fieldLabel: 'Molecules', fieldType: 'textarea', requiredField: true, placeholder: 'caffeine, CBD, dopamine' },
        { fieldLabel: 'Template', fieldType: 'dropdown', requiredField: true, fieldOptions: { values: [
          { option: 'Stacked (all methods)' }, { option: 'FTIR' }, { option: 'UV-Vis' },
          { option: 'Raman' }, { option: 'NMR' }, { option: 'PXRD' } ] } },
        { fieldLabel: 'YouTube visibility', fieldType: 'dropdown', requiredField: true, fieldOptions: { values: [
          { option: 'Draft (private)' }, { option: 'Unlisted' }, { option: 'Public (post)' } ] } },
      ] },
      options: {},
    },
    id: '6e8ca90b-0d2a-476c-9147-9d1b2d2141d0',
    name: 'Batch request (form)',
    type: 'n8n-nodes-base.formTrigger',
    typeVersion: 2.2,
    position: [-340, 200],
    webhookId: 'f1b2c3d4-0000-4000-8000-000000000001',
  },
  {
    parameters: {
      method: 'POST',
      url: 'https://api.anthropic.com/v1/messages',
      authentication: 'genericCredentialType',
      genericAuthType: 'httpHeaderAuth',
      sendHeaders: true,
      headerParameters: { parameters: [ { name: 'anthropic-version', value: '2023-06-01' } ] },
      sendBody: true,
      specifyBody: 'json',
      jsonBody: parseBodyExpr,
      options: {},
    },
    id: '3af7c601-683a-4f50-ba66-d4018b234534',
    name: 'AI: parse molecules',
    type: 'n8n-nodes-base.httpRequest',
    typeVersion: 4.2,
    position: [-100, 200],
    credentials: { httpHeaderAuth: ANTHROPIC_CRED },
  },
  {
    parameters: { mode: 'runOnceForAllItems', jsCode: parseListCode },
    id: 'a3d34979-87cf-41c7-9ba6-e0b175ed7994',
    name: 'Parse list',
    type: 'n8n-nodes-base.code',
    typeVersion: 2,
    position: [140, 200],
  },
  {
    parameters: {
      method: 'POST',
      url: 'https://api.anthropic.com/v1/messages',
      authentication: 'genericCredentialType',
      genericAuthType: 'httpHeaderAuth',
      sendHeaders: true,
      headerParameters: { parameters: [ { name: 'anthropic-version', value: '2023-06-01' } ] },
      sendBody: true,
      specifyBody: 'json',
      jsonBody: metaBodyExpr,
      options: {},
    },
    id: '049c9638-48fb-4d47-9851-979befd3b6c0',
    name: 'AI: metadata',
    type: 'n8n-nodes-base.httpRequest',
    typeVersion: 4.2,
    position: [380, 200],
    credentials: { httpHeaderAuth: ANTHROPIC_CRED },
  },
  {
    parameters: { mode: 'runOnceForEachItem', jsCode: buildMetaCode },
    id: '7cd60400-3ea1-4d07-9a12-ae0f21dc0b37',
    name: 'Build metadata',
    type: 'n8n-nodes-base.code',
    typeVersion: 2,
    position: [620, 200],
  },
  {
    parameters: {
      method: 'POST',
      url: 'http://host.docker.internal:8765/ensure',
      sendHeaders: true,
      headerParameters: { parameters: [ { name: 'X-Token', value: RENDER_TOKEN } ] },
      sendBody: true,
      specifyBody: 'json',
      // async: kick off the render (or instant-stage if it already exists), get a job id
      jsonBody: '={{ JSON.stringify({ scene_class: $json.scene_class, async: true }) }}',
      options: {},
    },
    id: 'a4e33ec8-b75c-4ad8-a38e-af3792ce38c3',
    name: 'Render: start',
    type: 'n8n-nodes-base.httpRequest',
    typeVersion: 4.2,
    position: [860, 200],
  },
  {
    parameters: {
      method: 'GET',
      url: "=http://host.docker.internal:8765/jobs/{{ $('Render: start').item.json.job_id }}",
      sendHeaders: true,
      headerParameters: { parameters: [ { name: 'X-Token', value: RENDER_TOKEN } ] },
      options: {},
    },
    id: 'b5f44fd9-c86d-4be9-b49f-c0840df49b04',
    name: 'Render: poll',
    type: 'n8n-nodes-base.httpRequest',
    typeVersion: 4.2,
    position: [1080, 200],
  },
  {
    parameters: {
      conditions: {
        options: { caseSensitive: true, leftValue: '', typeValidation: 'loose', version: 2 },
        conditions: [ { id: 'c-err-1', leftValue: '={{ $json.status }}', rightValue: 'error',
          operator: { type: 'string', operation: 'equals' } } ],
        combinator: 'and',
      },
      options: {},
    },
    id: 'c1111111-1111-4111-8111-111111111111',
    name: 'Render failed?',
    type: 'n8n-nodes-base.if',
    typeVersion: 2.2,
    position: [1300, 200],
  },
  {
    parameters: {
      errorMessage: "={{ 'Render failed for ' + $json.scene_class + ': ' + ($json.error || 'unknown') }}",
    },
    id: 'd2222222-2222-4222-8222-222222222222',
    name: 'Stop (render error)',
    type: 'n8n-nodes-base.stopAndError',
    typeVersion: 1,
    position: [1520, 60],
  },
  {
    parameters: {
      conditions: {
        options: { caseSensitive: true, leftValue: '', typeValidation: 'loose', version: 2 },
        conditions: [ { id: 'c-ready-1', leftValue: '={{ $json.status }}', rightValue: 'ready',
          operator: { type: 'string', operation: 'equals' } } ],
        combinator: 'and',
      },
      options: {},
    },
    id: 'e3333333-3333-4333-8333-333333333333',
    name: 'Render ready?',
    type: 'n8n-nodes-base.if',
    typeVersion: 2.2,
    position: [1520, 320],
  },
  {
    parameters: { amount: 15, unit: 'seconds' },
    id: 'f4444444-4444-4444-8444-444444444444',
    name: 'Wait 15s',
    type: 'n8n-nodes-base.wait',
    typeVersion: 1.1,
    position: [1300, 460],
    webhookId: 'a9b8c7d6-0000-4000-8000-000000000002',
  },
  {
    parameters: {
      operation: 'read',
      fileSelector: '=/incoming/{{ $json.incoming_name }}',
      options: { dataPropertyName: 'data' },
    },
    id: 'afa712a1-5006-4368-8509-6f11bbb17e74',
    name: 'Read MP4',
    type: 'n8n-nodes-base.readWriteFile',
    typeVersion: 1.1,
    position: [1740, 320],
  },
  {
    parameters: {
      resource: 'video',
      operation: 'upload',
      title: "={{ $('Build metadata').item.json.title }}",
      regionCode: 'US',
      categoryId: '27',
      binaryProperty: 'data',
      options: {
        privacyStatus: "={{ ({'Draft (private)':'private','Unlisted':'unlisted','Public (post)':'public'})[$('Batch request (form)').item.json['YouTube visibility']] || 'private' }}",
        description: "={{ $('Build metadata').item.json.description }}",
        tags: "={{ $('Build metadata').item.json.ytTags }}",
        selfDeclaredMadeForKids: false,
      },
    },
    id: '42fbf70a-e05b-487a-a432-48a8eba90ca8',
    name: 'Upload → YouTube',
    type: 'n8n-nodes-base.youTube',
    typeVersion: 1,
    position: [1980, 160],
    credentials: { youTubeOAuth2Api: YOUTUBE_CRED },
  },
  {
    parameters: {},
    id: '6dffbaba-fb97-4958-9886-7cdd0c5bdb1c',
    name: 'TikTok (scaffold — wire later)',
    type: 'n8n-nodes-base.noOp',
    typeVersion: 1,
    position: [1980, 340],
  },
  {
    parameters: {},
    id: '16d50160-0b19-4058-b983-65b898bedbf2',
    name: 'Reels (scaffold — wire later)',
    type: 'n8n-nodes-base.noOp',
    typeVersion: 1,
    position: [1980, 500],
  },
];

const connections = {
  'Batch request (form)': { main: [[{ node: 'AI: parse molecules', type: 'main', index: 0 }]] },
  'AI: parse molecules':  { main: [[{ node: 'Parse list', type: 'main', index: 0 }]] },
  'Parse list':           { main: [[{ node: 'AI: metadata', type: 'main', index: 0 }]] },
  'AI: metadata':         { main: [[{ node: 'Build metadata', type: 'main', index: 0 }]] },
  'Build metadata':       { main: [[{ node: 'Render: start', type: 'main', index: 0 }]] },
  'Render: start':        { main: [[{ node: 'Render: poll', type: 'main', index: 0 }]] },
  'Render: poll':         { main: [[{ node: 'Render failed?', type: 'main', index: 0 }]] },
  'Render failed?':       { main: [
    [{ node: 'Stop (render error)', type: 'main', index: 0 }],   // true  = errored
    [{ node: 'Render ready?', type: 'main', index: 0 }],         // false = keep checking
  ] },
  'Render ready?':        { main: [
    [{ node: 'Read MP4', type: 'main', index: 0 }],              // true  = file staged
    [{ node: 'Wait 15s', type: 'main', index: 0 }],              // false = still rendering
  ] },
  'Wait 15s':             { main: [[{ node: 'Render: poll', type: 'main', index: 0 }]] },  // loop
  'Read MP4':             { main: [[
    { node: 'Upload → YouTube', type: 'main', index: 0 },
    { node: 'TikTok (scaffold — wire later)', type: 'main', index: 0 },
    { node: 'Reels (scaffold — wire later)', type: 'main', index: 0 },
  ]] },
};

const workflow = {
  id: 'batchShorts01',
  name: 'Batch shorts — molecules → multi-platform',
  active: false,
  settings: { executionOrder: 'v1' },
  nodes,
  connections,
};

const out = path.join(__dirname, 'batch-shorts.json');
fs.writeFileSync(out, JSON.stringify(workflow, null, 2));
console.log('wrote ' + out);
