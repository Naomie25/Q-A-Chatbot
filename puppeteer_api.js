const express = require('express');
const puppeteer = require('puppeteer');

const app = express();
const PORT = 3000;

app.use(express.json());

app.post('/ask', async (req, res) => {
  const query = req.body.query;

  if (!query) {
    return res.status(400).json({ error: 'Missing field "query"' });
  }

  console.log('🔎 Received query:', query);

  let browser;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    const page = await browser.newPage();

    // User-Agent "normal"
    await page.setUserAgent(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    );

    // On va directement sur Wikipedia plutôt que Google
    const searchUrl =
      'https://en.wikipedia.org/wiki/Special:Search?go=Go&search=' +
      encodeURIComponent(query);

    console.log('🌍 Navigating to:', searchUrl);

    await page.goto(searchUrl, { waitUntil: 'networkidle2' });
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Récupérer le premier paragraphe un peu propre
    const snippet = await page.evaluate(() => {
      // Si on est redirigé directement sur un article :
      const paragraphs = document.querySelectorAll('#mw-content-text p');
      for (const p of paragraphs) {
        const text = p.innerText.trim();
        if (text.length > 50) {
          return text.slice(0, 500); // on limite un peu
        }
      }
      // Fallback : tout le texte de la page, tronqué
      if (document.body && document.body.innerText) {
        return document.body.innerText.trim().slice(0, 500);
      }
      return null;
    });

    if (!snippet) {
      console.log('⚠️ No snippet found on Wikipedia for query:', query);
      return res.json({ answer: 'No snippet found on Wikipedia.' });
    }

    console.log('✅ Snippet found, length:', snippet.length);
    return res.json({ answer: snippet });
  } catch (error) {
    console.error('❌ Error in /ask:', error);
    return res.status(500).json({ error: error.message });
  } finally {
    if (browser) {
      await browser.close();
    }
  }
});

app.listen(PORT, () => {
  console.log(`Puppeteer server listening on http://localhost:${PORT}`);
});

