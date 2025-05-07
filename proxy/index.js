// proxy/index.js
import express from 'express';
import fetch from 'node-fetch';

const app = express();
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

app.get('/proxy', async (req, res) => {
  const target = req.query.url;
  if (!target) return res.status(400).send('Missing url param');
  try {
    const binanceRes = await fetch(target);
    const data = await binanceRes.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.toString() });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Proxy listening on port ${port}`));
