// Rebuild the static stylesheet after editing classes in the HTML:
//   cd apps/web && npx tailwindcss@3 -c tailwind.config.js -i tailwind.input.css -o tailwind.css --minify
module.exports = { content: ["./index.html", "./whitepaper.html"], theme: { extend: {} }, plugins: [] };
