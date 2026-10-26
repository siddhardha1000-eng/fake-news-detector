"""
Dataset Collection and Curation Module
Collects, generates, and formats balanced datasets of Real and Fake news.
Supports downloading public benchmark datasets and compiling balanced corpora.
"""

import os
import random
import urllib.request
import pandas as pd
from typing import Optional


# Public benchmark mirrors (ISOT / WELFake subset mirrors or raw repositories)
BENCHMARK_URLS = {
    # Public small mirror for demonstration if online
    "sample_mirror": "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/sample_news.csv"
}


def create_comprehensive_dataset(output_path: str = "data/news_dataset.csv", n_samples: int = 1200) -> pd.DataFrame:
    """
    Constructs a rich, balanced news dataset containing both authentic (REAL=0)
    and fabricated/clickbait (FAKE=1) news across politics, science, health,
    technology, and general world affairs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Core authentic journalistic templates and samples (REAL news: label 0)
    real_news_sources = [
        "Reuters", "Associated Press", "BBC World News", "Nature Journal",
        "The Wall Street Journal", "Scientific American", "Bloomberg"
    ]

    real_articles_seed = [
        # Politics & Governance
        {
            "title": "Senate Approves Bipartisan Infrastructure Spending Package After Months of Negotiations",
            "text": "WASHINGTON (Reuters) - The United States Senate on Tuesday passed a bipartisan federal infrastructure package in a 69-30 vote. The legislation allocates 550 billion dollars in new federal spending for bridges, public transit, clean water networks, and broadband expansion over the next five years. 'This agreement proves that democratic institutions can still deliver tangible results for the American workforce,' said the Senate Majority Leader during a press briefing. The bill now heads to the House of Representatives for final consideration.",
            "category": "politics",
            "label": 0
        },
        {
            "title": "European Union Reaches Agreement on Stricter Carbon Border Emissions Tariffs",
            "text": "BRUSSELS (AP) - European Union member states and lawmakers finalized an agreement establishing the world's first carbon border adjustment mechanism. The framework requires importers of steel, cement, aluminum, and fertilizers to purchase certificates corresponding to the carbon emissions generated during production. EU climate commissioner stated that the tariff ensures domestic manufacturers are not disadvantaged by non-EU competitors operating under weaker environmental regulations.",
            "category": "politics",
            "label": 0
        },
        {
            "title": "Federal Reserve Holds Interest Rates Steady, Citing Moderating Inflation and Labor Market Stability",
            "text": "NEW YORK (Bloomberg) - The Federal Reserve opted to maintain its benchmark interest rate target range between 5.25% and 5.50% following its two-day policy summit. Federal Reserve Chair Jerome Powell stated during the press conference that while inflation has decelerated significantly from its peak, policymakers require continued empirical evidence before initiating rate reductions. Financial markets responded favorably to the central bank's measured policy stance.",
            "category": "politics",
            "label": 0
        },
        {
            "title": "Supreme Court Rules Unanimously on Regulatory Authority Over Navigable Inland Waters",
            "text": "WASHINGTON (AP) - The Supreme Court issued a unanimous opinion delineating federal oversight guidelines regarding wetland conservation. The justices ruled that the Environmental Protection Agency must establish a continuous surface connection between protected wetlands and traditional navigable bodies of water. Legal scholars noted the ruling provides greater clarity for agricultural and land development interests while defining statutory boundaries.",
            "category": "politics",
            "label": 0
        },
        # Science & Astronomy
        {
            "title": "James Webb Space Telescope Identifies Atmospheric Carbon Dioxide on Distant Exoplanet",
            "text": "BALTIMORE (Scientific American) - Astronomers utilizing high-resolution transmission spectroscopy from the James Webb Space Telescope have confirmed clear molecular signatures of carbon dioxide in the atmosphere of exoplanet WASP-39b. The gas giant, orbiting a star 700 light-years away, provides essential data on chemical evolution and planet formation mechanisms outside the Solar System. Findings were published in Nature by the international consortium.",
            "category": "science",
            "label": 0
        },
        {
            "title": "Fusion Energy Experiment Sustains Record Thermal Energy Output at UK Facility",
            "text": "OXFORD (Reuters) - Researchers at the Joint European Torus (JET) facility in Oxfordshire have achieved a record 59 megajoules of sustained fusion energy using deuterium-tritium plasma. The reaction maintained stable confinement for five seconds, demonstrating critical scaling validation for the larger ITER reactor currently under construction in Saint-Paul-les-Durance, France. The result marks a milestone toward commercially viable clean nuclear fusion.",
            "category": "science",
            "label": 0
        },
        {
            "title": "Paleontologists Discover Well-Preserved Cretaceous Dinosaur Fossil with Intact Skin Impressions",
            "text": "CALGARY (Nature) - A field expedition in the Badlands of Alberta has uncovered an extraordinarily preserved hadrosaur fossil displaying three-dimensional epidermal soft tissue impressions. The specimen dates back approximately 75 million years. Analysis indicates specialized scales designed for moisture retention and thermal regulation, providing fresh insights into late Cretaceous terrestrial ecology.",
            "category": "science",
            "label": 0
        },
        # Health & Medicine
        {
            "title": "Clinical Trial Results Confirm Significant Cognitive Slowing in Early Stage Alzheimer Treatment",
            "text": "BOSTON (Reuters) - A randomized double-blind Phase 3 clinical trial involving 1,800 participants demonstrated that monoclonal antibody therapy reduced cognitive and clinical decline by 27% compared to placebo over an 18-month duration. Lead investigator Dr. Christopher Chen emphasized that while amyloid plaque clearance was verified via PET scans, monitoring for microhemorrhages remains essential. The findings were presented at the Clinical Trials on Alzheimer's Disease symposium.",
            "category": "health",
            "label": 0
        },
        {
            "title": "World Health Organization Endorses New Cost-Effective Malaria Vaccine Following African Trials",
            "text": "GENEVA (BBC) - The World Health Organization has officially recommended the programmatic deployment of the R21/Matrix-M malaria vaccine for infants residing in high-transmission regions. Clinical trials across Burkina Faso, Kenya, Mali, and Tanzania recorded a 75% reduction in symptomatic infections. The serum can be produced affordably at high volume, closing critical immunization supply shortages across sub-Saharan Africa.",
            "category": "health",
            "label": 0
        },
        {
            "title": "Researchers Map Gut Microbiome Metabolic Pathways Linked to Cardiovascular Health",
            "text": "BOSTON (AP) - A peer-reviewed study published in the New England Journal of Medicine identified specific bacterial metabolites that modulate arterial inflammation and lipid oxidation. Researchers analyzed dietary intake and serum biomarkers across a longitudinal cohort of 12,000 subjects over a decade, concluding that dietary fiber fermentation produces short-chain fatty acids that directly protect vascular endothelium.",
            "category": "health",
            "label": 0
        },
        # Technology & Economy
        {
            "title": "Semiconductor Manufacturer Inaugurates Advanced Silicon Fabrication Facility in Arizona",
            "text": "PHOENIX (Reuters) - A major global semiconductor foundry celebrated the opening of its multi-billion dollar advanced semiconductor fabrication plant in Phoenix, Arizona. The facility will produce next-generation 3-nanometer microchips for mobile computing, artificial intelligence accelerators, and automotive safety systems. The company announced the hiring of 3,000 engineering and technical specialists in cooperation with local universities.",
            "category": "technology",
            "label": 0
        },
        {
            "title": "Open Source Consortium Releases Lightweight AI Model Optimized for Mobile Hardware",
            "text": "SAN FRANCISCO (TechCrunch) - The Open Source AI Foundation announced the public release of a distilled 3-billion-parameter neural network capable of performing complex reasoning and code generation locally on handheld devices without cloud server connectivity. The architecture features optimized 4-bit quantization, achieving 45 tokens per second on consumer smartphone processors.",
            "category": "technology",
            "label": 0
        },
        {
            "title": "International Cyber Security Agency Issues Guidance on Quantum-Resistant Encryption Migration",
            "text": "LONDON (BBC) - The National Cyber Security Centre published standardized migration frameworks for public institutions and banking sectors to adopt post-quantum cryptographic standards approved by NIST. The recommendations urge organizations to phase out legacy RSA and elliptic curve algorithms before 2030 to prevent retroactive decryption of intercepted communications.",
            "category": "technology",
            "label": 0
        }
    ]

    # Core fabricated / clickbait / conspiracy templates (FAKE news: label 1)
    fake_articles_seed = [
        # Politics & Conspiracies
        {
            "title": "SHOCKING: Leaked Pentagon Documents Reveal Secret Weather Control Weapon Caused Worldwide Earthquakes",
            "text": "A brave whistleblower from deep inside the Pentagon has just leaked explosive top secret documents proving what truth-seekers have suspected for decades! Hidden underground military facilities have been using high frequency electromagnetic waves to trigger earthquakes and hurricanes on command to manipulate foreign elections. Mainstream media refuses to cover this absolute bombshell revelation! Share this before the global elites delete it from the internet permanently!",
            "category": "politics",
            "label": 1
        },
        {
            "title": "BREAKING: Globalist Leaders Caught on Hot Mic Admitting All Paper Currency Will Be Terminated by Midnight",
            "text": "In an urgent broadcast leaked by patriotic insiders, world financial elites were caught admitting during an unrecorded closed-door meeting that all cash is being invalidated by next Friday! You will wake up to zero dollars in your bank accounts unless you immediately transfer all your assets into special digital tracking tokens run by central banks. Wake up people, the total enslavement program is starting right now! Do not trust the corrupt corrupt media!",
            "category": "politics",
            "label": 1
        },
        {
            "title": "CONFIRMED: Voting Machines in All 50 States Were Programmed with Secret Remote Satellite Control Chip",
            "text": "An anonymous military intelligence commander has come forward with undeniable proof that voting booths across the entire nation were secretly equipped with secret satellite receiving antennas designed by foreign intelligence agents. According to the whistleblower, millions of ballots were flipped in real-time from an offshore yacht in international waters! Judges refuse to look at the evidence because they are terrified of the deep state cabal!",
            "category": "politics",
            "label": 1
        },
        {
            "title": "BOMBSHELL: White House Secret Tunnel Discovered Connecting to Offshore Submarine Base",
            "text": "Construction workers fixing a broken water main near Washington DC stumbled upon a 400-mile underground tunnel complex equipped with bullet trains and secret nuclear bunkers. Sources confirm that top politicians use this tunnel to secretly escape to offshore mega-yachts whenever national crises occur. The government is threatening anyone who speaks about this with immediate arrest! Share this truth far and wide!",
            "category": "politics",
            "label": 1
        },
        # Health & Medical Hoaxes
        {
            "title": "MIRACLE CURE: Doctors Terrified After Grandma Cures All Stage 4 Cancers Using Dandelion Water and Baking Soda",
            "text": "Big Pharma is scrambling to hide this simple everyday kitchen recipe! An 82-year-old grandmother from Ohio has completely cured over 500 cancer patients in just three days using nothing but boiled dandelion roots and two tablespoons of ordinary baking soda! Greedy pharmaceutical executives and corrupt oncologists tried to sue her because this miracle 2-dollar home cure destroys their multi-billion dollar chemotherapy monopoly overnight. Read the secret formula before it gets taken down!",
            "category": "health",
            "label": 1
        },
        {
            "title": "WARNING: New Cellular Towers Are Radiating Microscopic Nanobots That Rewrite Human DNA While You Sleep",
            "text": "Holistic health researchers have uncovered terrifying microscopic proof: 5G and 6G cellular communication frequencies are not communication towers at all, but vibrational activation transmitters for microscopic synthetic parasites floating in municipal drinking water! Once energized by cell towers, these bio-chips alter your neurological cortex and allow remote control of your thoughts! Protect your home with aluminum shielding immediately before it is too late!",
            "category": "health",
            "label": 1
        },
        {
            "title": "EXPOSED: Common Lemon Juice Mixture Can Reverse Age by 30 Years in 14 Days, Doctors Spew Lies",
            "text": "You won't believe what cosmetic surgeons don't want you to find out! A secret tribal extract combined with fresh squeezed lemon juice and salt has been scientifically proven by independent researchers to regrow failing organs and erase 30 years of aging in under two weeks. The medical establishment wants you sick and aging so you buy expensive prescriptions. Drink this magical tonic every morning on an empty stomach to become completely immune to every known disease!",
            "category": "health",
            "label": 1
        },
        # Science & Tech Hoaxes
        {
            "title": "NASA ADMISSION: Massive Ancient Pyramids and Artificial Roads Photographed on Dark Side of Moon",
            "text": "A renegade NASA engineer who recently escaped the space agency has uploaded 4,000 uncensored satellite photographs showing monumental stone pyramids, glass domes, and active mining complexes on the far side of the Moon! NASA has been keeping this supreme cosmic secret hidden from humanity for over 60 years while spending trillions of taxpayer dollars on black-budget extraterrestrial treaties. The truth cannot be suppressed any longer!",
            "category": "science",
            "label": 1
        },
        {
            "title": "GENIUS INVENTOR: Backyard Mechanic Builds Free Energy Generator from Scrap Magnets, Mysteriously Disappears",
            "text": "A humble auto repairman in Michigan discovered how to tap into zero-point cosmic vacuum energy using six neodymium magnets and a ceiling fan motor, producing infinite free electricity for entire neighborhoods without paying a dime to electric utility companies! Within 48 hours of posting his blueprint on social media, mysterious black government SUVs raided his house and confiscated all prototypes. Share this viral blueprint before it's erased forever!",
            "category": "science",
            "label": 1
        },
        {
            "title": "TECH TITAN SCANDAL: Silicon Valley Ceo Unveils Secret AI Chip Designed to Read Brainwaves from 500 Miles Away",
            "text": "In a leaked leaked video from a private billionaire retreat, a prominent tech mogul boasted that their new satellite network can decode human thoughts, emotional states, and unspoken passwords directly from space! The artificial intelligence analyzes the bio-magnetic vibrations of your skull and sends targeted advertisements into your subconscious dreams while you rest. The globalists are celebrating total mental surveillance!",
            "category": "technology",
            "label": 1
        }
    ]

    # Systematic linguistic variations & topical expansions to create a realistic, robust corpus
    real_topics = [
        ("Treasury Department", "quarterly economic growth indicators", "fiscal policy targets and labor productivity improvements"),
        ("United Nations Climate Summit", "renewable energy investments", "multinational carbon emission reduction targets"),
        ("Pharmaceutical Research Board", "immunotherapy oncology trials", "published peer-reviewed clinical survival statistics"),
        ("European Space Agency", "deep space probe observations", "telemetry data from Jupiter's icy moons"),
        ("Federal Aviation Administration", "commercial aviation safety audits", "implementation of upgraded radar avionics"),
        ("International Monetary Fund", "global trade volume projections", "emerging market currency resilience metrics"),
        ("Department of Energy", "next-generation battery storage systems", "grid stability trials across municipal substations"),
        ("World Meteorological Organization", "atmospheric pressure variations", "long-range satellite climate monitoring datasets"),
        ("National Institutes of Health", "genomic sequencing for hereditary disorders", "cohort verification across medical research centers"),
        ("Global Agricultural Council", "drought-resistant crop cultivars", "yield assessments in arid farming communities")
    ]

    fake_topics = [
        ("Deep State Syndicate", "secret weather control satellites", "mind-control frequencies hidden inside public Wi-Fi routers"),
        ("Brave Truth-Seeking Whistleblower", "miracle herbal detox formula", "how greedy pharmaceutical cartels cover up natural cures"),
        ("Secret Underground Base Workers", "cloned politicians replacing world leaders", "unbelievable proof that mainstream media is hiding"),
        ("Anonymous Military Insider", "alien starships parked behind the sun", "covert treaties between billionaires and extraterrestrials"),
        ("Banned Holistic Doctor", "how drinking boiled root tea dissolves all tumors", "urgent warnings before the medical board silences this truth"),
        ("Patriotic Citizen Investigative Reporter", "stolen elections managed by foreign submarine servers", "irrefutable proof that the entire monetary system is rigged"),
        ("Mysterious Rogue Physicist", "infinite free energy perpetual motion device", "how power utility monopolies threaten independent inventors"),
        ("Leaked Classified Video", "human thoughts intercepted by cellular towers", "nanotech mind control substances injected in bottled beverages"),
        ("Secret Vatican Archives Vault", "suppressed ancient prophecies about world takeover", "explosive truths that the elite will do anything to censor"),
        ("Shadow Government Defector", "secret underground city built for the super-rich", "imminent total collapse planned for tomorrow morning")
    ]

    records = []

    # Add seeds first
    for item in real_articles_seed:
        records.append(item)
    for item in fake_articles_seed:
        records.append(item)

    # Generate synthetic realistic articles to reach desired sample size
    target_each = n_samples // 2

    real_templates = [
        ("Official Report: {org} Announces Findings on {topic}",
         "According to an official statement released by {org}, recent assessments on {topic} indicate positive developments. Officials reported: '{detail}.' The research was conducted across multiple independent monitoring stations over a two-year evaluation period. Technical documentation is scheduled for publication in next month's public registry."),
        
        ("International Summit: {org} Outlines New Regulatory Framework for {topic}",
         "Delegates representing forty member states convened at the annual summit to formalize standard operating guidelines concerning {topic}. A spokesperson for {org} stated that the framework emphasizes verified empirical benchmarks and transparent reporting: '{detail}.' Industry analysts welcomed the announcement as a stabilizing measure."),
        
        ("Scientific Publication: New Analysis on {topic} Released by {org}",
         "Researchers affiliated with {org} have presented peer-reviewed findings regarding {topic}. The empirical investigation demonstrated reproducible results under controlled experimental conditions. 'The quantitative indicators confirm {detail},' noted the lead author. Further longitudinal trials will commence in the coming fiscal quarter."),

        ("Economic Bulletin: {org} Reports Latest Market Metrics Concerning {topic}",
         "In its quarterly statistical review, {org} outlined comprehensive economic indicators evaluating {topic}. The report underscored steady structural resilience across core supply chains, noting that '{detail}.' Central bank representatives signaled adherence to prudent monetary guidelines.")
    ]

    fake_templates = [
        ("URGENT ALERT: {org} Exposes Terrifying Secret Regarding {topic}!",
         "You won't believe what mainstream fake news is hiding from the public! An anonymous insider connected to {org} has just leaked catastrophic revelations about {topic}. According to undeniable documents: '{detail}!' Share this immediately on all platforms before government agents scrub this webpage from the internet! Wake up before it is too late!"),

        ("BOMBSHELL EVIDENCE: {org} Caught In Massive Cover-up of {topic}!",
         "The truth has finally exploded into the open! Corrupt global elites and media puppets tried their best to silence anyone who spoke up about {topic}, but a heroic patriotic defector from {org} has brought forth ironclad proof: '{detail}.' Spread the word to family and friends! The corrupt establishment is panicking!"),

        ("LEAKED DOCUMENTS: What {org} Refuses to Tell You About {topic}",
         "Shocking undercover footage has caught high-ranking figures from {org} admitting what they really think about {topic}. The whistleblowers claim: '{detail}!' Corrupt big tech algorithms are already shadowbanning this story. Read and save this report before it gets censored forever!"),

        ("EXPOSED: The Dark Reality Behind {topic} That {org} Doesn't Want You To Know",
         "Independent researchers are sounding the alarm over terrifying developments concerning {topic}. While corrupt news outlets claim everything is safe, insider sources within {org} reveal: '{detail}!' Protect yourself and your family right now with this vital knowledge! They can't stop the truth!")
    ]

    # Expand Real Articles
    while len([r for r in records if r["label"] == 0]) < target_each:
        org, topic, detail = random.choice(real_topics)
        t_headline, t_body = random.choice(real_templates)
        title = t_headline.format(org=org, topic=topic)
        text = t_body.format(org=org, topic=topic, detail=detail)
        records.append({
            "title": title,
            "text": text,
            "category": "general",
            "label": 0
        })

    # Expand Fake Articles
    while len([r for r in records if r["label"] == 1]) < target_each:
        org, topic, detail = random.choice(fake_topics)
        t_headline, t_body = random.choice(fake_templates)
        title = t_headline.format(org=org, topic=topic)
        text = t_body.format(org=org, topic=topic, detail=detail)
        records.append({
            "title": title,
            "text": text,
            "category": "general",
            "label": 1
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to '{output_path}'.")
    print(f"Total rows: {len(df)} | Real (0): {(df['label'] == 0).sum()} | Fake (1): {(df['label'] == 1).sum()}")
    return df


def load_dataset(filepath: str = "data/news_dataset.csv") -> pd.DataFrame:
    """
    Loads dataset from disk. If not found, generates the comprehensive balanced dataset.
    """
    if not os.path.exists(filepath):
        print(f"Dataset file '{filepath}' not found. Generating default dataset...")
        return create_comprehensive_dataset(filepath)
    df = pd.read_csv(filepath)
    print(f"Loaded dataset '{filepath}' with {len(df)} records.")
    return df


if __name__ == "__main__":
    create_comprehensive_dataset("data/news_dataset.csv", n_samples=1200)
