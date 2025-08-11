const express = require('express');
const puppeteer = require('puppeteer');

const app = express();
const PORT = 3000;

app.get('/ask', async (req, res) => {
  const query = req.query.q;
  if (!query) {
    return res.status(400).json({ error: 'Missing query parameter "q"' });
  }

  try {
    const browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    const page = await browser.newPage();

    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
    await page.goto(searchUrl, { waitUntil: 'domcontentloaded' });

    // Essayer de récupérer le snippet de réponse Google
    // Selector souvent utilisé pour le résumé en haut :
    // div[data-attrid="wa:/description"] ou div[data-tts="answers"] ou div[jsname="W297wb"]

    const snippet = await page.evaluate(() => {
      // Plusieurs sélecteurs possibles, on essaie dans l’ordre

      const selectors = [
        'div[data-attrid="wa:/description"]',
        'div[data-tts="answers"]',
        'div[jsname="W297wb"]',
        'div[data-attrid="kc:/location/location:short_description"]',
        'div[data-attrid="kc:/people/person:short_description"]',
        '.V3FYCf', // autre snippet possible
        '.hgKElc'  // snippet dans knowledge panel
      ];

      for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el && el.innerText.trim().length > 0) {
          return el.innerText.trim();
        }
      }
      // fallback : récupérer le premier paragraphe textuel dans le résultat de recherche
      const firstResult = document.querySelector('.g .VwiC3b');
      if (firstResult) {
        return firstResult.innerText.trim();
      }
      return null;
    });

    await browser.close();

    if (snippet) {
      res.json({ answer: snippet });
    } else {
      // fallback si pas trouvé
      res.json({ answer: "No direct snippet found, try another query." });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Puppeteer server listening on http://localhost:${PORT}`);
});
