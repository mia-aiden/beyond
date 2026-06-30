"""
Source:
https://www.comparativeagendas.net/codebook
https://www.comparativeagendas.net/pages/master-codebook
"""

CAP_TOPIC_PROTOTYPES = [
    {
        "topic_id": 1,
        "name": "Macroeconomics",
        "prototype_text": (
            "Macroeconomics: issues related to general domestic macroeconomic policy. "
            "Subtopics include inflation, cost of living, prices, interest rates, the "
            "unemployment rate, monetary policy, central banks, the treasury, public debt, "
            "budgeting, deficit reduction, tax policy and tax enforcement, manufacturing "
            "policy, industrial revitalization and growth, and wage or price controls."
        ),
    },
    {
        "topic_id": 2,
        "name": "Civil Rights",
        "prototype_text": (
            "Civil Rights: issues related to civil rights and minority rights. "
            "Subtopics include minority, ethnic, and racial discrimination, sex, gender, "
            "and sexual orientation discrimination, age discrimination, disability and "
            "disease discrimination, voting rights and the franchise, freedom of speech, "
            "religious freedom, freedom of expression, privacy rights, access to government "
            "information, abortion rights, and anti-government activity groups."
        ),
    },
    {
        "topic_id": 3,
        "name": "Health",
        "prototype_text": (
            "Health: issues related to health care policy and health systems. "
            "Subtopics include health care reform, health insurance reform and cost, "
            "pharmaceuticals and medical devices, medical facilities, provider and insurer "
            "payments, medical liability and malpractice, health care labor supply and "
            "licensing, disease prevention and treatment, children's health, mental health, "
            "long-term care, prescription drug coverage and cost, tobacco, alcohol, and "
            "illegal drug abuse, and health research and development."
        ),
    },
    {
        "topic_id": 4,
        "name": "Agriculture",
        "prototype_text": (
            "Agriculture: issues related to agriculture policy. "
            "Subtopics include agricultural foreign trade, subsidies to farmers and ranchers, "
            "food inspection and safety, food marketing and promotion, animal and crop disease, "
            "pesticide regulation, fisheries and fishing, and agricultural research and development."
        ),
    },
    {
        "topic_id": 5,
        "name": "Labor",
        "prototype_text": (
            "Labor: issues related to labor, employment, and pensions. "
            "Subtopics include worker safety and compensation, job training and workforce "
            "development, employee benefits, pensions, retirement accounts, labor unions, "
            "collective bargaining, minimum wage, overtime compensation, labor law, youth "
            "employment, child labor, and migrant, guest, and seasonal workers."
        ),
    },
    {
        "topic_id": 6,
        "name": "Education",
        "prototype_text": (
            "Education: issues related to education policy. "
            "Subtopics include higher education, student loans, education finance, colleges "
            "and universities, elementary and secondary school reform, school safety, efforts "
            "to improve educational standards and outcomes, education for underprivileged "
            "students, adult literacy, bilingual education, rural education, vocational "
            "education, special education, and education research and development."
        ),
    },
    {
        "topic_id": 7,
        "name": "Environment",
        "prototype_text": (
            "Environment: issues related to environmental policy. "
            "Subtopics include drinking water safety and supply, wastewater and solid waste "
            "disposal, hazardous waste and toxic chemicals, air pollution, climate change, "
            "noise pollution, recycling and resource conservation, indoor environmental hazards, "
            "species and forest protection, endangered species, wildlife protection, and "
            "environmental technology research and development."
        ),
    },
    {
        "topic_id": 8,
        "name": "Energy",
        "prototype_text": (
            "Energy: issues related to energy policy. "
            "Subtopics include nuclear energy and nuclear waste, electricity and utilities, "
            "natural gas and oil, drilling, oil spills, flaring, oil and gas prices, coal, "
            "alternative and renewable energy, biofuels, hydrogen, geothermal power, energy "
            "conservation, energy efficiency, and energy research and development."
        ),
    },
    {
        "topic_id": 9,
        "name": "Immigration",
        "prototype_text": (
            "Immigration: issues related to immigration, refugees, and citizenship."
        ),
    },
    {
        "topic_id": 10,
        "name": "Transportation",
        "prototype_text": (
            "Transportation: issues related to transportation policy. "
            "Subtopics include mass transportation, highways, road construction and safety, "
            "air travel and aviation safety, airports and air traffic control, railroads and "
            "rail freight, maritime transportation and shipping, waterways and channels, "
            "infrastructure and public works, and transportation research and development."
        ),
    },
    {
        "topic_id": 12,
        "name": "Law and Crime",
        "prototype_text": (
            "Law and Crime: issues related to law, crime, courts, policing, and family issues. "
            "Subtopics include law enforcement agencies, white collar crime, fraud, cyber-crime, "
            "illegal drugs and drug trafficking, court administration, prisons and parole, "
            "juvenile crime, child abuse and exploitation, domestic violence, child welfare, "
            "family law, criminal and civil codes, crime prevention and control, police, and "
            "domestic security responses to terrorism."
        ),
    },
    {
        "topic_id": 13,
        "name": "Social Welfare",
        "prototype_text": (
            "Social Welfare: issues related to social welfare policy. "
            "Subtopics include poverty assistance for low-income families, food assistance, "
            "welfare dependency programs, tax credits for low-income families, elderly issues "
            "and government pensions, disability assistance, volunteer associations, charities, "
            "youth organizations, parental leave, and child care."
        ),
    },
    {
        "topic_id": 14,
        "name": "Housing",
        "prototype_text": (
            "Housing: issues related to housing and urban affairs. "
            "Subtopics include housing and community development, neighborhood development, "
            "national housing policy, urban development, rural housing, rural development, "
            "housing affordability and low-income housing, public housing projects, housing for "
            "veterans, housing for the elderly, homelessness, and housing research and development."
        ),
    },
    {
        "topic_id": 15,
        "name": "Domestic Commerce",
        "prototype_text": (
            "Domestic Commerce: issues related to domestic commerce, banking, finance, and business "
            "regulation. Subtopics include banking and non-bank financial institutions, securities "
            "and commodities trading, consumer finance, mortgages, credit cards, insurance regulation, "
            "bankruptcy, corporate mergers and antitrust, corporate governance, small businesses, "
            "copyrights and patents, domestic disaster relief, tourism, consumer fraud and safety, "
            "and sports regulation, gambling, and personal fitness."
        ),
    },
    {
        "topic_id": 16,
        "name": "Defense",
        "prototype_text": (
            "Defense: issues related to defense and military policy. "
            "Subtopics include defense alliances, security assistance, intelligence and espionage, "
            "military readiness, nuclear weapons and proliferation, military aid, military personnel "
            "and dependents, military courts, procurement, weapons systems, military installations, "
            "reserve forces, military hazardous waste, homeland security, civilian personnel in the "
            "defense industry, military contractors, direct war-related foreign operations, and "
            "claims against the military."
        ),
    },
    {
        "topic_id": 17,
        "name": "Technology",
        "prototype_text": (
            "Technology: issues related to space, science, technology, and communications. "
            "Subtopics include government use of space, space exploration, military use of space, "
            "commercial space development, science and technology transfer, telecommunications, "
            "high-speed internet infrastructure, newspapers, publishing, radio, broadcast television, "
            "weather forecasting, oceanography, geological surveys, the computer industry, internet "
            "regulation, computer security, and technology research and development."
        ),
    },
    {
        "topic_id": 18,
        "name": "Foreign Trade",
        "prototype_text": (
            "Foreign Trade: issues related to foreign trade policy. "
            "Subtopics include trade negotiations, disputes, and agreements, tax treaties, export "
            "regulation and subsidies, international private business investment, business "
            "competitiveness, balance of payments issues, tariffs, import barriers, import regulation, "
            "and exchange rates."
        ),
    },
    {
        "topic_id": 19,
        "name": "International Affairs",
        "prototype_text": (
            "International Affairs: issues related to international affairs and foreign aid. "
            "Subtopics include foreign aid, international resource exploitation and law of the sea, "
            "developing countries, international finance, the World Bank and IMF, sovereign debt, "
            "Western Europe and the European Union, specific foreign countries or regions, human "
            "rights, genocide and crimes against humanity, international organizations such as the "
            "United Nations and International Criminal Court, international terrorism, diplomacy, "
            "embassies, citizens abroad, visas, and passports."
        ),
    },
    {
        "topic_id": 20,
        "name": "Government Operations",
        "prototype_text": (
            "Government Operations: issues related to government operations and administrative matters. "
            "Subtopics include intergovernmental relations, bureaucracy and oversight, postal service, "
            "civil employees and civil service, nominations and appointments, currency and national mints, "
            "government procurement and contractors, government property management, tax administration, "
            "public scandals and impeachment, branch relations, constitutional reforms, regulation of "
            "political campaigns, campaign finance, voter registration, census and statistics, the "
            "capital city, claims against the government, and national holidays."
        ),
    },
    {
        "topic_id": 21,
        "name": "Public Lands",
        "prototype_text": (
            "Public Lands: issues related to public lands, water management, and territorial issues. "
            "Subtopics include national parks, memorials, historic sites, recreation, indigenous affairs "
            "and indigenous lands, forest management, natural resources, forest fires, livestock grazing, "
            "water resources, flood control, territorial and dependency issues, and devolution."
        ),
    },
]
