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

/**
 * @param {string} link
 * @param {puppeteer.Browser} browser
 */
const scrapIdentifiersFromSpecificVolumeUrl = async (link, browser) => {
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

        await page.goto(link, {
            referer: "https://www.google.com/",
            waitUntil: "networkidle2",
        });

        return page.evaluate(
            (link, allowedTypes) => {
                const piis = [];
                let prevLink = "";
                let nextLink = "";

                const navElement = document.querySelector(
                    "nav.issue-navigation",
                );

                if (navElement instanceof HTMLElement) {
                    const prevA = navElement.querySelector(".navigation-pre a");
                    if (prevA instanceof HTMLAnchorElement) {
                        prevLink = prevA.href;
                    }

                    const nextA =
                        navElement.querySelector(".navigation-next a");
                    if (nextA instanceof HTMLAnchorElement) {
                        nextLink = nextA.href;
                    }
                }

                const PREFIX_1 = "/science/article/pii/";
                const PREFIX_2 =
                    "https://www.sciencedirect.com/science/article/pii/";

                const articleList = document.querySelectorAll(
                    ".article-list > li, .js-article-list > li",
                );

                for (const article of articleList) {
                    if (!(article instanceof HTMLLIElement)) {
                        console.error(`could not parse article title`);
                        continue;
                    }

                    const subtypeContainer = article.querySelector(
                        ".js-article-subtype",
                    );

                    if (!(subtypeContainer instanceof HTMLElement)) {
                        console.error("subtype container not html element");
                        continue;
                    }

                    const subtype = subtypeContainer.innerText
                        .trim()
                        .toUpperCase();

                    if (!allowedTypes.includes(subtype)) {
                        console.warn("article not in correct subtype");
                        continue;
                    }

                    const articleHeading = article.querySelector("h3");

                    if (!(articleHeading instanceof HTMLHeadingElement)) {
                        console.error("article heading not an html element");
                        continue;
                    }

                    const links = articleHeading.querySelectorAll("a");

                    if (links.length === 0) {
                        console.error("no links found in page");
                    }

                    for (const link of links) {
                        if (link.href.startsWith(PREFIX_1)) {
                            piis.push(link.href.substring(PREFIX_1.length));
                        } else if (link.href.startsWith(PREFIX_2)) {
                            piis.push(link.href.substring(PREFIX_2.length));
                        } else {
                            console.error("could not parse pii from href");
                        }
                    }
                }

                return {
                    prevLink: prevLink,
                    currLink: link,
                    nextLink: nextLink,
                    piis: piis,
                };
            },
            link,
            ["RESEARCH ARTICLE", "REVIEW ARTICLE"],
        );
    } catch (error) {
        console.log(error);
        throw error;
    }
};

/**
 *
 * @typedef {object} MongoUrlOptions
 * @property {string} [username] - The unique identifier for the file. Required.
 * @property {string} [password] - The title of the content. Optional.
 * @property {string} [host] - A brief summary or abstract of the content. Optional.
 * @property {number} [port] - A specific highlight or key takeaway from the content. Optional.
 * @property {string} [database] - A specific highlight or key takeaway from the content. Optional.
 *
 * @param {MongoUrlOptions} options
 */
const createMongoUrl = (options) => {
    let URL = "mongodb://";
    if (options.username && options.password) {
        URL += options.username + ":" + options.password + "@";
    }
    URL += (options.host || "localhost") + ":";
    URL += (options.port || "27017") + "/";
    URL += options.database || "admin";
    return URL;
};

/**
 * @param {string} seed
 */
const getJournalNameVolume = (seed) => {
    const JOURNAL_PREFIX = "https://www.sciencedirect.com/journal/";
    const parts = seed.substring(JOURNAL_PREFIX.length).split("/");
    return {
        journalName: parts[0],
        journalVolume: parts[2],
    };
};

/**
 * @param {string} volumeUrl
 * @param {puppeteer.Browser} browser
 */
const crawlJournalVolumes = async (volumeUrl, browser) => {
    const MAX_ERROR_TRY = 3;
    let seed = volumeUrl;
    const { journalVolume } = getJournalNameVolume(volumeUrl);
    const startVolume = journalVolume;
    let lastVolume = journalVolume;
    let errTry = 0;

    /** @type {Map<string, string[]>} */
    const piis = new Map();

    while (seed !== "") {
        logger.info(`starting crawing of ${seed}`);

        try {
            const { journalVolume } = getJournalNameVolume(seed);
            const e = await scrapIdentifiersFromSpecificVolumeUrl(
                seed,
                browser,
            );
            logger.info(JSON.stringify(e, null, 4));

            const existingList = piis.get(journalVolume) ?? [];
            piis.set(journalVolume, [...existingList, ...e.piis]);

            lastVolume = journalVolume;

            seed = e.prevLink;
            errTry = 0;
        } catch (e) {
            logger.error(e.stack);
            errTry++;
            if (errTry == MAX_ERROR_TRY) {
                logger.error(`max retries hit for seed ${seed}, breaking`);
                break;
            }
        }

        await new Promise((r) => setTimeout(r, 5000 + Math.random() * 3000));
    }

    return {
        startVolume,
        lastVolume,
        piis: Object.fromEntries(piis),
    };
};

const dumpJournalPIIs = async () => {
    try {
        logger.info("Loading dotnet");
        dotenv.config({
            path: ["../.env"],
        });

        logger.info("Connecting to Browser API...");

        const browser = await puppeteer.launch({
            browser: "chrome",
            executablePath: "/snap/bin/chromium",
            headless: true,
            // browserWSEndpoint: BROWSER_WS,
        });

        logger.info("mongo client creation started");

        const mongo = new MongoClient(
            createMongoUrl({
                username: process.env["MONGO_USERNAME"],
                password: process.env["MONGO_PASSWORD"],
                host: process.env["MONGO_HOST"],
                port: parseInt(process.env["MONGO_PORT"] || "27017"),
                database: process.env["MONGO_DATABASE"],
            }),
            {
                retryWrites: true,
                writeConcern: {
                    w: "majority", // Acknowledge from majority of replica set members
                    j: true, // Ensure write is committed to the journal
                    wtimeout: 5000, // Timeout after 5 seconds if write concern is not met
                },
                // https://notes.dt.in.th/MongoDBAuthSource
                authSource: "admin",
            },
        );

        await mongo.connect();

        logger.info("mongo client created");

        const volBase = [
            "https://www.sciencedirect.com/journal/expert-systems-with-applications-x/vol/1/suppl/C",
            // "https://www.sciencedirect.com/journal/burnout-research/vol/1/issue/2",
        ];

        const DUMP_COLLECTION_NAME = "VOLUMES-SCRAP";
        const col = mongo.db().collection(DUMP_COLLECTION_NAME);

        for (const vol of volBase) {
            try {
                const { journalName, journalVolume } =
                    getJournalNameVolume(vol);

                const h = await col.findOne({
                    journalName: journalName,
                });

                if (h !== null) {
                    logger.warn(
                        `skipping over journal name ${journalName} as already present in mongo`,
                    );
                    // TODO: add logic to check if the current volume is lesser than the last volume updated volume
                    continue;
                }

                logger.info(`starting extraction for journal ${vol}`);
                const d = await crawlJournalVolumes(vol, browser);
                logger.info(
                    `extraction finished for journal ${vol}\n${JSON.stringify(d.piis, null, 4)}`,
                );

                await col.replaceOne(
                    { journalName: journalName },
                    {
                        journalName: journalName,
                        ...d,
                    },
                    { upsert: true },
                );

                logger.info(
                    `saved extracted data for ${journalName} into mongodb`,
                );
            } catch (e) {
                logger.error(e.stack);
            }
        }

        await browser.close();
        await mongo.close();
    } catch (e) {
        logger.error(e.stack);
    }
};
