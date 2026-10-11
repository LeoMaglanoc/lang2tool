import fs from 'node:fs';
const css = await (await fetch('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap')).text();
const urls = [...new Set(css.match(/https:\/\/fonts\.gstatic\.com\/[^)\s]+/g))];
fs.mkdirSync('public/assets/fonts',{recursive:true});
let local = css;
for (const [i,url] of urls.entries()) {
  const filename = `font-${i}.ttf`;
  const response = await fetch(url); if(!response.ok) throw new Error('Font download failed');
  fs.writeFileSync('public/assets/fonts/'+filename,Buffer.from(await response.arrayBuffer()));
  local=local.replaceAll(url,`/lang2tool/assets/fonts/${filename}`);
}
fs.writeFileSync('src/fonts.css',local);
console.log('Vendored fonts:',urls.length);
