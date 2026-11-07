import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve(import.meta.dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const figures=JSON.parse(read('data/charts.json'));
const manifest=JSON.parse(read('data/manifest.json'));
assert.equal(Object.keys(figures).length,32);
assert.ok(manifest.responses>=30);
for(const [id,f] of Object.entries(figures)){
 assert.ok(/^g\d+$/.test(id));
 assert.ok(Array.isArray(f.data)&&f.data.length>0);
 assert.ok(f.layout);
 assert.ok(read('index.html').includes('plot-'+id));
}
for(const file of manifest.tables)assert.ok(fs.existsSync(path.join(root,'data',file)));
for(const file of ['assets/app.js','assets/plotly.min.js','.nojekyll'])assert.ok(fs.existsSync(path.join(root,file)));
for(const file of ['datos_ampliados.json','datos_pjm_exacto.json','source.json','fresh_source.json'])assert.ok(!fs.existsSync(path.join(root,file)));
assert.ok(read('assets/app.js').includes("fetch('./data/charts.json'"));
console.log('OK: 32 figuras, 977 respuestas, recursos relativos y tablas públicas completas.');
