"""Hand-written answers that follow each process, used only to check the checklists."""

GOLD = {
    "engineering": [
        ("Build a command line expense tracker.", """Requirements:
- Add an expense with an amount, category and date.
- List expenses and show the monthly total.
- Save the expense data in a local file.

Design:
A storage module reads and writes expenses as JSON. A report module adds up totals, and a cli module handles commands.

expense-tracker/
  README.md
  requirements.txt
  .gitignore
  src/cli.py
  src/storage.py
  src/report.py
  tests/test_storage.py
  tests/test_report.py

Setup:
1. Create a virtual environment.
2. Install dependencies with pip install -r requirements.txt.
3. Run the tests with pytest.
4. Run the tracker with python src/cli.py.
"""),
        ("Create a URL shortener web service.", """Goals:
- Accept a long URL and return a short code.
- Redirect each short code to its original URL.
- Count how many visits each link receives.

Architecture:
An Express router handles the requests. A links module creates codes, and a store module keeps the URL records.

url-shortener/
  README.md
  package.json
  .gitignore
  src/server.js
  src/links.js
  src/store.js
  tests/links.test.js
  tests/store.test.js

Getting started:
1. Clone the repository.
2. Install packages with npm install.
3. Run the checks with npm test.
4. Run the service with npm start.
"""),
        ("Make a file backup utility in Rust.", """Requirements:
- Copy changed files from a source folder to a backup folder.
- Keep a log of every backup that runs.
- Skip files listed in an ignore file.

Design:
A scan module finds changed files. A copy module writes them, and a log module records each backup run.

backup-tool/
  README.md
  Cargo.toml
  .gitignore
  src/main.rs
  src/scan.rs
  src/copy.rs
  tests/test_scan.rs
  tests/test_copy.rs

Setup:
1. Install the Rust toolchain.
2. Create a release build with cargo build.
3. Run the tests with cargo test.
4. Run a backup with cargo run.
"""),
        ("Set up a recipe manager application.", """Features:
- Save recipes with ingredients and cooking steps.
- Search recipes by name or ingredient.
- Export a shopping list for chosen recipes.

Components:
A models module defines recipes. A search module finds matches, and an export module writes the shopping list.

recipe-manager/
  README.md
  pyproject.toml
  .gitignore
  app/models.py
  app/search.py
  app/export.py
  tests/test_search.py
  tests/test_export.py

Installation:
1. Create a virtual environment.
2. Install the package with pip install -e .
3. Run the tests with pytest.
4. Run the app with python -m app.
"""),
    ],
    "story": [
        ("Write a story about a lighthouse keeper who finds a message in a bottle.",
         "Old Tomas was a lighthouse keeper who lived on a small island far from any town. "
         "One grey morning he found a bottle on the rocks, and the message inside said a sailor was lost and needed help. "
         "Tomas tried to signal the mainland with his radio, but the storm had broken the antenna. "
         "He tried again with the great lamp, flashing it all night, but no ship answered. "
         "Then he decided to row out alone and search the reef for the sailor. "
         "The waves grew taller and pushed his little boat toward the rocks. "
         "Suddenly he saw a broken mast and a man clinging to it in the dark water. "
         "Tomas pulled hard on the oars and dragged the lost sailor into the boat. "
         "Together they returned to the island, and the sailor was finally safe. "
         "From then on, Tomas kept that message in the lighthouse to remember the sailor he saved."),
        ("Tell a story about a girl whose kite is stolen by a crow.",
         "Lena was a cheerful girl who lived in a windy village beside a green hill. "
         "One afternoon a huge crow swooped down, and her red kite was stolen from her hands. "
         "Lena tried to chase the crow across the fields, but it flew too high. "
         "She searched the old barn for the kite, but found only feathers and dust. "
         "Then she decided to climb the tall oak where the crow had built its nest. "
         "The branches shook in the wind as she climbed higher and higher. "
         "Suddenly the crow dived at her, and she faced it with a piece of bread in her hand. "
         "The crow took the bread and dropped the red kite beside her. "
         "Lena carried the kite home, happy and relieved. "
         "From then on, she left bread for the crow, and her kite was never stolen again."),
    ],
    "teaching": [
        ("Teach fractions to a Primary 2 student.",
         "Today we will learn about fractions and how to share things fairly. "
         "Do you remember when we cut a pizza at the party? "
         "First, a fraction tells us about equal parts of one whole thing. "
         "Next, we count how many equal parts there are. "
         "Then we count the parts that we take. "
         "Finally, we write the two numbers as a fraction. "
         "For example, imagine one apple cut into two equal parts. "
         "Let me show you how I write one half on the board. "
         "Now it is your turn to try it with this paper circle. "
         "Fold it into two equal parts and color one part. "
         "What fraction of the circle did you color? "
         "Great work with your paper circle today. "
         "Today we learned that a fraction shows equal parts of a whole."),
        ("Explain the water cycle to a young Primary 3 learner.",
         "Today we will learn about the water cycle and where rain comes from. "
         "Have you ever seen a puddle dry up after the rain? "
         "First, the warm sun heats the water in the sea. "
         "Next, the water rises into the sky as tiny drops. "
         "Then the drops join together and make clouds. "
         "Finally, the clouds get heavy and the water falls as rain. "
         "For example, imagine steam going up from a hot cup. "
         "Let me show you how I draw the water cycle with arrows. "
         "Now it is your turn to try drawing the arrows on your page. "
         "Start at the sea and follow the water to the cloud. "
         "What happens to the water when a cloud gets heavy? "
         "You drew a very clear water cycle today. "
         "Today we learned that water moves from the sea to the sky and back again."),
    ],
}
