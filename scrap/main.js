const puppeteer = require("puppeteer");

/**
 * @param {puppeteer.Page} page
 */
const processPage = (page) => {
  return page.$eval("#abstracts", (container) => {
    /**
     * @param {Element | null} node
     */
    const nodeHasText = (node) => {
      return (
        node instanceof HTMLDivElement ||
        node instanceof HTMLParagraphElement ||
        node instanceof HTMLSpanElement
      );
    };

    // Fetch the sub-elements from the previously fetched container element
    // Get the displayed text and return it (`.innerText`)
    const highlightNodeList =
      container
        ?.querySelector(".author-highlights")
        ?.querySelectorAll('.react-xocs-list-item [id^="par"]') || [];

    const highlight = [...highlightNodeList]
      .filter(nodeHasText)
      .map((n) => n.innerText)
      .map((h) => {
        h = h.trim();
        return h.endsWith(".") ? h : h + ".";
      })

      .join(" ");

    const abstractNodeList =
      container
        ?.querySelector(".abstract.author")
        ?.querySelectorAll('[id^="spar"]') || [];

    const abstract = [...abstractNodeList]
      .filter(nodeHasText)
      .map((n) => n.innerText)
      .join(" ");

    return { highlight, abstract };
  });
};

/**
 * @param {string[]} filenames
 */
async function run(filenames) {
  try {
    console.log("Connecting to Browser API...");

    const browser = await puppeteer.launch({
      browser: "chrome",
      executablePath: "/snap/bin/chromium",
      headless: true,
      // browserWSEndpoint: BROWSER_WS,
    });

    for (const filename of filenames) {
      const scienceDirectUri = `https://www.sciencedirect.com/science/article/abs/pii/${filename}`;
      const waybackMachineUri = `https://web.archive.org/web/20250512000000/${scienceDirectUri}`;
      console.log(waybackMachineUri);
      const page = await browser.newPage();

      const requestHeaders = {
        dnt: "1",
        pragma: "no-cache",
        priority: "u=0, i",
        connection: "keep-alive",
        "sec-ch-ua":
          '"Chromium";v="136", "Brave";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "cross-site",
        "sec-fetch-user": "?1",
        "sec-gpc": "1",
        "upgrade-insecure-requests": "1",
        accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
      };

      // intercept the request with the custom headers
      await page.setExtraHTTPHeaders({ ...requestHeaders });
      await page.setJavaScriptEnabled(false);
      await page.setUserAgent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
      );

      page.setRequestInterception(true);
      page.on("request", (request) => {
        if (request.resourceType() === "script") request.abort();
        else request.continue();
      });

      await page.goto(waybackMachineUri, {
        referer: "https://www.google.com/",
        waitUntil: "networkidle2",
      });

      // CAPTCHA handling: If you're expecting a CAPTCHA on the target page, use the following code snippet to check the status of Browser API's automatic CAPTCHA solver
      // const client = await page.createCDPSession();
      // console.log('Waiting captcha to solve...');
      // const { status } = await client.send('Captcha.waitForSolve', {
      //   detectTimeout: 10000,
      // });
      // console.log('Captcha solve status:', status);
      await page.waitForSelector("#abstracts", {
        timeout: 2000,
      });

      const errorExists = await page
        .$eval("#errorBorder #error", (el) => el !== null)
        .catch((e) => {
          // This catch block handles the case where the selector #errorBorder #error is not found at all
          // or any other error during the $eval.
          // If the selector is not found, $eval throws. We want to treat this as "no error element found".
          console.log("Error element not found on page, proceeding normally.");
          return false; // No error element found
        });

      let waitTime = 4000;

      if (errorExists) {
        console.log(
          "Error detected on the initial page. Navigating to fallback page..."
        );

        await page.goto(scienceDirectUri, {
          referer: "https://www.google.com/",
          waitUntil: "networkidle2",
        });

        await page.waitForSelector("#abstracts", {
          timeout: 2000,
        });

        waitTime = 15000;
      }

      const point = await processPage(page);
      console.log(point);
      await new Promise((r) => setTimeout(r, waitTime + Math.random() * 2000));
    }

    await browser.close();
  } catch (error) {
    console.log(error);
  }
}

run(["S0001457518304810"]);
