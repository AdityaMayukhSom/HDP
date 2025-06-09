const puppeteer = require("puppeteer");
const dotenv = require("dotenv");
const winston = require("winston");
const { Pool } = require("pg");
const { MongoClient } = require("mongodb");

require("winston-daily-rotate-file");

const transportAll = new winston.transports.DailyRotateFile({
    filename: "application-%DATE%.log",
    dirname: "log/all",
    datePattern: "YYYY-MM-DD-HH-mm",
    frequency: "30m",
    maxSize: "1m",
    maxFiles: "14d",
});

const transportError = new winston.transports.DailyRotateFile({
    level: "error",
    dirname: "log/error",
    filename: "application-error-%DATE%.log",
    datePattern: "YYYY-MM-DD-HH-mm",
    zippedArchive: true,
    frequency: "30m",
    maxSize: "1m",
    maxFiles: "14d",
});

// @ts-ignore
const logger = winston.createLogger({
    transports: [
        //
        // - Write all logs with importance level of `error` or higher to `error.log`
        //   (i.e., error, fatal, but not other levels)
        //
        transportError,
        //
        // - Write all logs with importance level of `info` or higher to `combined.log`
        //   (i.e., fatal, error, warn, and info, but not trace)
        //
        transportAll,
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple(),
            ),
        }),
    ],
});

const REQUEST_HEADERS = {
    dnt: "1",
    pragma: "no-cache",
    priority: "u=0, i",
    connection: "keep-alive",
    "sec-ch-ua": '"Chromium";v="136", "Brave";v="136", "Not.A/Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "cross-site",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.7",
    "cache-control": "no-cache",
};

const USER_AGENT =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

const getIndianTime = () => {
    return new Date().toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata",
    });
};

/**
 * Removes all invisible characters from a string, preserves ASCII special characters,
 * and replaces various Unicode dash-like characters with a plain ASCII hyphen-minus.
 *
 * @param {string} text The input string, which can contain any Unicode characters.
 * @returns {string} The cleaned string.
 */
function cleanStringRemoveInvisibleChars(text) {
    if (typeof text !== "string") {
        throw new TypeError("Input must be a string.");
    }

    if (text.trim().length === 0) {
        return "";
    }

    // Step 1: Replace various Unicode dash-like characters with a plain ASCII hyphen-minus (-)
    // This regex covers common Unicode dashes, hyphens, and minus signs.
    // U+2010 (HYPHEN), U+2011 (NON-BREAKING HYPHEN), U+2012 (FIGURE DASH),
    // U+2013 (EN DASH), U+2014 (EM DASH), U+2015 (HORIZONTAL BAR),
    // U+2212 (MINUS SIGN)
    let processedText = text.replace(/[\u2010-\u2015\u2212]/g, "-");

    // Step 2: Remove invisible characters.
    // This regex targets:
    // - ASCII control characters (0x00-0x1F, 0x7F)
    // - Unicode control/formatting characters (various ranges, including zero-width spaces,
    //   soft hyphens, byte order marks, etc.)
    // It explicitly does not remove printable ASCII characters (0x20-0x7E),
    // which includes all standard letters, numbers, and special symbols like !@#$%^&*()
    // It also preserves common whitespace characters like \n, \r, \t.
    const invisibleCharsPattern = new RegExp(
        "[\x00-\x1F\x7F]" + // ASCII control characters and DEL
            "|[\u00ad\u034f\u17b4\u17b5\u180e\u200b-\u200f\u2028-\u202f\u2060-\u2064\u206a-\u206f\ufeff\ufff9-\ufffc]",
        "g", // Global flag to replace all occurrences
    );
    processedText = processedText.replace(invisibleCharsPattern, "");

    return processedText;
}

/**
 * @param {string} pii
 * @param {puppeteer.Page} page
 */
const processPage = (pii, page) => {
    return page.evaluate((pii) => {
        let title = null;
        let titleRoot = document.querySelector("#screen-reader-main-title");
        if (!titleRoot) {
            console.warn(
                `elem with id 'screen-reader-main-title' not found for ${pii}`,
            );
            titleRoot = document?.querySelector("h1");
        }
        if (titleRoot instanceof HTMLElement) {
            const titleCont = titleRoot.querySelector(".title-text");
            if (titleCont instanceof HTMLElement) {
                console.info(
                    `extracting title from title container (inner tag) in ${pii}`,
                );
                title = titleCont.innerText.trim();
            } else {
                console.info(
                    `extracting title from title root (outer tag) in ${pii}`,
                );
                title = titleRoot.innerText.trim();
            }
        } else {
            console.error(`could not found title root in ${pii}`);
        }

        // Fetch the sub-elements from the previously fetched container element
        // Get the displayed text and return it (`.innerText`)
        const container = document.querySelector("#abstracts");
        const hlRoot = container?.querySelector(".author-highlights .list");

        // const hlSel = [
        //   `.list .react-xocs-list-item .list-contents`,
        //   `.list .react-xocs-list-item *:not(.list-label)`,
        //   `.list .react-xocs-list-item`,
        //   `.list`,
        // ];

        let highlight = "";
        if (hlRoot instanceof HTMLElement) {
            highlight = hlRoot.innerText
                .trim()
                .replaceAll("•", ". ")
                .replaceAll("\n", "");

            if (highlight.startsWith(".")) {
                highlight = highlight.slice(1) + ".";
            }

            highlight = highlight.replace(/\.(\s|\.)*\./g, ". ").trim();
        }

        // let highlightNodeList = [];
        // for (const sel of hlSel) {
        //   let tmpHlList = hlRoot?.querySelectorAll(sel);
        //   if (tmpHlList !== undefined && tmpHlList.length > 0) {
        //     // @ts-ignore
        //     highlightNodeList = tmpHlList;
        //     break;
        //   }
        // }

        // let highlight =
        //   highlightNodeList.length > 0
        //     ? [...highlightNodeList]
        //         .filter((n) => n instanceof HTMLElement)
        //         .map((n) => n.innerText)
        //         .map((h) => {
        //           h = h
        //             .trim()
        //             .replaceAll("•", ".")
        //             .replaceAll("\n", "")
        //             .replaceAll(/[^\x00-\x7F]/g, "");
        //           return h.endsWith(".") ? h : h + ".";
        //         })
        //         .join(" ")
        //     : hlRoot instanceof HTMLElement
        //       ? hlRoot.innerText.trim().replaceAll("•", "").replaceAll("\n", "")
        //       : "";

        // if (highlightNodeList.length > 0) {
        //   if (highlight.startsWith("Highlights")) {
        //     highlight = highlight.replace("Highlights", "");
        //   }

        //   highlight = highlight.replaceAll("\n", "");
        // }

        // const abstractNodeList =
        //   container?.querySelector(".abstract")?.querySelectorAll('[id^="spar"]') ||
        //   [];

        // const abstract = [...abstractNodeList]
        //   .filter(nodeHasText)
        //   .map((n) => n.innerText)
        //   .join(" ");

        let abstract = "";
        const absRoot = container?.querySelector(".author");

        if (absRoot instanceof HTMLElement) {
            const headingChildren = absRoot.querySelectorAll(
                ".section-title, h1, h2, h3",
            );

            for (const children of headingChildren) {
                children.parentNode?.removeChild(children);
            }

            abstract = absRoot.innerText
                // .replaceAll(/[^\x00-\x7F]/g, "")
                .replaceAll(/[\uFFFD]|\p{C}/gu, "")
                .trim();
        }

        // const highlightContainer = container?.querySelector(".author-highlights");

        // const highlight = nodeHasText(highlightContainer)
        // ? highlightContainer.innerText
        // : null;

        return {
            title,
            highlight,
            abstract,
        };
    }, pii);
};

/**
 * @param {string[]} piis
 * @param {puppeteer.Browser} browser
 * @param {boolean} directlyFromSD
 */
const scrapDataFromPIIs = async (piis, browser, directlyFromSD = false) => {
    try {
        const page = await browser.newPage();

        // intercept the request with the custom headers
        await page.setExtraHTTPHeaders({ ...REQUEST_HEADERS });
        await page.setJavaScriptEnabled(false);
        await page.setUserAgent(USER_AGENT);

        page.setRequestInterception(true);

        page.on("request", (request) => {
            if (
                request.resourceType() === "script" ||
                request.resourceType() === "font" ||
                request.resourceType() === "stylesheet" ||
                request.resourceType() === "image"
            )
                request.abort();
            else request.continue();
        });

        page.on("console", async (msg) => {
            const msgArgs = msg.args();
            for (let i = 0; i < msgArgs.length; ++i) {
                logger.log(msg.type(), await msgArgs[i].jsonValue());
            }
        });

        /**
         * @typedef {object} DataType
         * @property {string} pii - The unique identifier for the file. Required.
         * @property {string | null} title - The title of the content. Optional.
         * @property {string | null} abstract - A brief summary or abstract of the content. Optional.
         * @property {string | null} highlight - A specific highlight or key takeaway from the content. Optional.
         */

        /**
         * @type {DataType[]}
         */
        const data = [];

        for (const pii of piis) {
            const scienceDirectUri = `https://www.sciencedirect.com/science/article/abs/pii/${pii}`;
            const waybackMachineUri = `https://web.archive.org/web/20250512000000/${scienceDirectUri}`;

            let errorInWaybackMachine = false;

            if (!directlyFromSD) {
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
                // await page.waitForSelector("#abstracts", {
                //   timeout: 2000,
                // });

                const errorExists = await page
                    .$eval(
                        "#errorBorder #error, .error-card",
                        (el) => el !== null,
                    )
                    .catch((e) => {
                        // This catch block handles the case where the selector #errorBorder #error is not found at all
                        // or any other error during the $eval.
                        // If the selector is not found, $eval throws. We want to treat this as "no error element found".
                        logger.info(
                            `no error element not found on page for file ${pii}, proceeding normally.`,
                        );
                        return false; // No error element found
                    });

                const noHighlight = await page
                    .$eval(
                        "#abstracts .author-highlights .list",
                        (el) => el === null,
                    )
                    .catch((e) => {
                        logger.error(
                            `No highlight exists ${pii}, will go to science direct..`,
                        );
                        return true; // No error element found
                    });

                const noAbstract = await page
                    .$eval("#abstracts .author", (el) => el === null)
                    .catch((e) => {
                        logger.error(
                            `No abstract exists ${pii}, will go to science direct..`,
                        );
                        return true; // No error element found
                    });

                errorInWaybackMachine =
                    errorExists || noHighlight || noAbstract;
            }

            let waitTime = 3000;

            if (errorInWaybackMachine || directlyFromSD) {
                if (directlyFromSD) {
                    logger.info(
                        `Requested scraping directly from Science Direct for PII ${pii}.`,
                    );
                } else if (errorInWaybackMachine) {
                    logger.error(
                        `Error detected on the initial page for file ${pii}. Navigating to fallback page.`,
                    );
                }

                await page.goto(scienceDirectUri, {
                    referer: "https://www.google.com/",
                    waitUntil: "networkidle2",
                });

                waitTime = 4000;
            }

            try {
                const point = await processPage(pii, page);
                // logger.info(JSON.stringify(point, null, 4));
                data.push({ ...point, pii: pii });
            } catch (e) {
                logger.error(`error while processing page for pii ${pii}`, e);
                data.push({
                    pii: pii,
                    title: null,
                    abstract: null,
                    highlight: null,
                });
            }

            await new Promise((r) =>
                setTimeout(r, waitTime + Math.random() * 3000),
            );
        }

        return data;
    } catch (error) {
        logger.log(error);
        throw error;
    }
};

const main = async () => {
    dotenv.config({
        path: ["../.env"],
    });

    try {
        logger.info("Connecting to Browser API...");

        const browser = await puppeteer.launch({
            browser: "chrome",
            executablePath: "/snap/bin/chromium",
            headless: true,
            // browserWSEndpoint: BROWSER_WS,
        });

        // const h = [
        // "S0014488619302298",
        // "S0034528818315327",
        // "S0003347220302657",
        // "S000334722030289X",
        // "S0009279719320046",
        // "S0014483519306608",
        // "S0003347220303018",
        // "S000927971932004",
        // ];
        // const d = await scrapDataFromPIIs(h, browser, false);
        // console.log(d);
        // return;

        const pool = new Pool({
            user: process.env["PG_USERNAME"],
            password: process.env["PG_PASSWORD"],
            host: process.env["PG_HOST"],
            port: parseInt(process.env["PG_PORT"] || "5432"),
            database: process.env["PG_DATABASE"],
        });

        let updtCnt = 0;
        const batchSize = 4;

        /** @type {import("pg").QueryConfig<[number | string]>} */
        const piiQuery = {
            name: "fetch-batched-piis",
            text: `
            SELECT "PII"
            FROM "MixSub"
            WHERE LENGTH("OriginalAbstract") <= 600
                AND "BetterAbstract" IS NULL
                AND "IsProcessed" = FALSE
            GROUP BY "PII"
            ORDER BY LENGTH("OriginalAbstract") ASC
            LIMIT $1
            `,
            values: [batchSize],
        };

        /**
         * @type {import("pg").QueryConfig<[string, string, string, string]>}
         */
        const updateAbstractHighlightQuery = {
            name: "update-abstract-highlight",
            text: `
            UPDATE "MixSub"
            SET "Title" = $1,
                "BetterAbstract" = $2,
                "BetterHighlight" = $3,
                "IsProcessed" = TRUE
            WHERE "PII" = $4
            `,
        };

        /**
         * @type {import("pg").QueryConfig<[]>}
         */
        const todoQuery = {
            name: "todo-abstract-highlight",
            text: `
            SELECT COUNT("PII") AS "TODO"
            FROM "MixSub"
            WHERE LENGTH("OriginalAbstract") <= 600
                AND "BetterAbstract" IS NULL
                AND "IsProcessed" = FALSE;
            `,
        };

        const todoRes = await pool.query(todoQuery, []);
        const todoCnt = parseInt(todoRes.rows[0]["TODO"]);
        logger.info(`TOTAL VALUES TO UPDATE :: ${todoCnt}`);

        while (updtCnt < todoCnt) {
            const client = await pool.connect();

            const remRes = await pool.query(todoQuery, []);
            const remCnt = parseInt(remRes.rows[0]["TODO"]);
            logger.info(`REMAINING VALUES TO UPDATE :: ${remCnt}`);

            const res = await client.query(piiQuery);

            /** @type {string[]} */
            const piis = res.rows.map((r) => r["PII"]);
            if (piis.length == 0) {
                // we are breaking using update count, but this is just extra protection
                logger.info("breaking as piis count is zero");
                break;
            }

            // /** @type {string[]} */
            // const bhls = res.rows.map((r) => r["BetterHighlight"]);

            try {
                logger.info(
                    `batch started at ${getIndianTime()} with contents ${piis}`,
                );

                const data = await scrapDataFromPIIs(piis, browser, true);

                logger.info(JSON.stringify(data, null, 4));

                for (let i = 0; i < piis.length; ++i) {
                    if (!data[i].abstract === null) {
                        logger.error(
                            `found abstract to be falsy for ${piis[i]}`,
                        );
                        data[i].abstract = ""; // do not set to null, set to empty string
                    }

                    if (data[i].highlight === null) {
                        logger.error(
                            `found highlight to be falsy for ${piis[i]}`,
                        );
                        data[i].highlight = ""; // do not set to null, set to empty string
                    }

                    data[i].highlight = cleanStringRemoveInvisibleChars(
                        data[i].highlight || "",
                    );
                    data[i].abstract = cleanStringRemoveInvisibleChars(
                        data[i].abstract || "",
                    );

                    // logger.info(`Existing BetterHighlight\n\n${bhls[i]}`)

                    // const el = data[i].highlight?.length || 0;

                    // if (el < bhls[i].length) {
                    //   logger.error("extracted data length is half or even less");
                    // }

                    try {
                        await client.query(
                            updateAbstractHighlightQuery, //
                            [
                                data[i].title, //
                                data[i].abstract,
                                data[i].highlight,
                                piis[i],
                            ],
                        );
                    } catch (e) {
                        logger.error("could not update for pii", piis[i]);
                    } finally {
                        updtCnt += 1;
                    }
                }

                logger.info(
                    `batch finished at ${getIndianTime()} with contents ${piis}`,
                );
            } catch (e) {
                logger.error("error occurred for batch", piis, e);
                logger.error(e.stack);
            }

            client.release();
        }

        await browser.close();
    } catch (e) {
        logger.error("error occurred inside while loop for update count");
        logger.error(e.stack);
    }
};

main()
    .then(() => process.exit(0))
    .catch((e) => process.exit(1));
