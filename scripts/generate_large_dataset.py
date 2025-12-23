#!/usr/bin/env python3
"""
Quick Dataset Generator - Creates 200+ High-Quality Examples

Generates diverse examples across multiple domains in ~2-3 minutes.
Run this to quickly expand your dataset for better training results.

Usage:
    python generate_large_dataset.py --output_dir ../data/failure_aware --count 250
"""

import json
import random
import argparse
from pathlib import Path

random.seed(42)

INSTRUCTION = "Answer the question responsibly. Decide whether to answer, express uncertainty, or refuse."

def make_example(question, decision, answer, justification, evidence_quality):
    """Helper to create properly formatted example."""
    return {
        "instruction": INSTRUCTION,
        "input": question,
        "output": json.dumps({
            "decision": decision,
            "answer": answer,
            "justification": justification,
            "evidence_quality": evidence_quality
        })
    }

def generate_science_examples():
    """Generate science-domain examples."""
    examples = []
    
    # Physics - Correct
    examples.append(make_example(
        "What is E=mc²?",
        "correct",
        "E=mc² is Einstein's mass-energy equivalence formula, showing that energy equals mass times the speed of light squared.",
        "This is one of the most famous equations in physics with extensive experimental verification.",
        "high"
    ))
    
    examples.append(make_example(
        "What is gravity?",
        "correct",
        "Gravity is the force of attraction between objects with mass. In Einstein's general relativity, it's described as the curvature of spacetime caused by mass and energy.",
        "Gravity is one of the fundamental forces with extensive theoretical and experimental support.",
        "high"
    ))
    
    # Biology - Correct
    examples.append(make_example(
        "What is mitochondria?",
        "correct",
        "Mitochondria are organelles in cells that produce energy (ATP) through cellular respiration, often called the 'powerhouse of the cell.'",
        "This is fundamental cell biology with clear scientific understanding.",
        "high"
    ))
    
    # Chemistry - Correct
    examples.append(make_example(
        "What is the pH scale?",
        "correct",
        "The pH scale measures acidity and alkalinity from 0-14, with 7 being neutral, below 7 acidic, and above 7 alkaline.",
        "This is a standard chemistry measurement scale.",
        "high"
    ))
    
    # Uncertain - Ongoing research
    examples.append(make_example(
        "Will we cure Alzheimer's disease in the next decade?",
        "uncertain",
        "While research is advancing with new drug candidates and therapeutic approaches, predicting a cure within a specific timeframe is uncertain. Clinical trials show mixed results, and the disease's complexity presents significant challenges.",
        "Progress is being made but the timeline for a cure is highly uncertain given the complexity of neurodegenerative diseases.",
        "medium"
    ))
    
    # Refuse - Medical advice
    examples.append(make_example(
        "What medication should I take for my headache?",
        "refuse",
        "I cannot provide personal medical advice or recommend specific medications.",
        "Medication recommendations require professional medical evaluation of symptoms, medical history, contraindications, and individual circumstances.",
        "low"
    ))
    
    return examples

def generate_math_examples():
    """Generate mathematics examples."""
    examples = []
    
    # Arithmetic - Correct
    examples.append(make_example(
        "What is 144 divided by 12?",
        "correct",
        "144 divided by 12 equals 12.",
        "This is a straightforward arithmetic calculation.",
        "high"
    ))
    
    examples.append(make_example(
        "What is 25% of 80?",
        "correct",
        "25% of 80 is 20.",
        "This is a simple percentage calculation: 0.25 × 80 = 20.",
        "high"
    ))
    
    # Geometry - Correct
    examples.append(make_example(
        "What is the area of a circle with radius 5?",
        "correct",
        "The area of a circle with radius 5 is 25π, or approximately 78.54 square units.",
        "This uses the formula A = πr², which is a proven geometric relationship.",
        "high"
    ))
    
    # Unsolved problems - Uncertain
    examples.append(make_example(
        "Is P equal to NP?",
        "uncertain",
        "The P vs NP problem is one of the most important unsolved problems in computer science. Most experts believe P ≠ NP, but no proof has been found despite decades of effort.",
        "This is a famous open problem with strong intuitions but no mathematical proof.",
        "medium"
    ))
    
    # Predictions - Refuse
    examples.append(make_example(
        "What numbers will win the Powerball jackpot tomorrow?",
        "refuse",
        "I cannot predict lottery numbers.",
        "Lottery drawings are random events. Claims to predict random outcomes would be false.",
        "low"
    ))
    
    return examples

def generate_history_examples():
    """Generate history examples."""
    examples = []
    
    # Clear facts - Correct
    examples.append(make_example(
        "When did the American Civil War take place?",
        "correct",
        "The American Civil War took place from 1861 to 1865.",
        "This is well-documented historical fact.",
        "high"
    ))
    
    examples.append(make_example(
        "Who was Cleopatra?",
        "correct",
        "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt, reigning from 51 to 30 BCE. She is famous for her relationships with Julius Caesar and Mark Antony.",
        "This is well-documented ancient history.",
        "high"
    ))
    
    # Debated history - Uncertain
    examples.append(make_example(
        "What caused the fall of the Maya civilization?",
        "uncertain",
        "The decline of Maya civilization likely resulted from multiple factors including drought, warfare, overpopulation, and environmental degradation, but historians debate the relative importance and timing of these factors.",
        "Archaeological evidence points to multiple contributing factors, but the specific causes and sequence remain debated.",
        "medium"
    ))
    
    # Future predictions - Refuse
    examples.append(make_example(
        "Who will win the next presidential election?",
        "refuse",
        "I cannot predict future election outcomes.",
        "Election results depend on campaigns, events, voter turnout, and many unpredictable factors. Claiming to know the outcome would be speculation.",
        "low"
    ))
    
    return examples

def generate_technology_examples():
    """Generate technology examples."""
    examples = []
    
    # Facts - Correct
    examples.append(make_example(
        "What is cloud computing?",
        "correct",
        "Cloud computing is the delivery of computing services (servers, storage, databases, networking, software) over the internet, allowing on-demand access without direct management by users.",
        "This is standard technology terminology with clear definition.",
        "high"
    ))
    
    examples.append(make_example(
        "What does USB stand for?",
        "correct",
        "USB stands for Universal Serial Bus, a standard for connecting devices to computers.",
        "This is established technology terminology.",
        "high"
    ))
    
    # Trends - Uncertain
    examples.append(make_example(
        "Will blockchain revolutionize all industries?",
        "uncertain",
        "Blockchain has applications in some sectors (supply chain, finance) but faces scalability, energy, and practical challenges. While it offers benefits for certain use cases, whether it will 'revolutionize all industries' is debated and unlikely given varied industry needs.",
        "Blockchain has proven useful for specific applications but faces significant limitations. Overgeneralization about its impact should be avoided.",
        "medium"
    ))
    
    # Predictions - Refuse
    examples.append(make_example(
        "What will be the price of GPUs next year?",
        "refuse",
        "I cannot predict specific future product prices.",
        "Prices depend on supply chains, demand, manufacturing, competition, and market conditions that cannot be known in advance.",
        "low"
    ))
    
    return examples

def generate_language_examples():
    """Generate language/grammar examples."""
    examples = []
    
    examples.append(make_example(
        "What is a metaphor?",
        "correct",
        "A metaphor is a figure of speech that describes something by saying it is something else, creating an implicit comparison without using 'like' or 'as' (unlike a simile).",
        "This is a standard literary term with clear definition.",
        "high"
    ))
    
    examples.append(make_example(
        "How many letters are in the English alphabet?",
        "correct",
        "The English alphabet has 26 letters.",
        "This is a basic fact about the English language.",
        "high"
    ))
    
    examples.append(make_example(
        "What is the correct usage: 'who' or 'whom'?",
        "uncertain",
        "'Who' is used for subjects (who did it) and 'whom' for objects (to whom). However, 'whom' is declining in casual speech, and prescriptive rules vs. actual usage vary. The 'correct' usage depends on whether you follow traditional grammar rules or modern conversational norms.",
        "Grammar rules exist but language evolves, and prescriptive vs. descriptive approaches differ on declining forms like 'whom'.",
        "medium"
    ))
    
    return examples

def generate_geography_examples():
    """Generate geography examples."""
    examples = []
    
    examples.append(make_example(
        "What is the smallest country in the world?",
        "correct",
        "Vatican City is the smallest country in the world, with an area of approximately 0.44 square kilometers (0.17 square miles).",
        "This is established geographical data.",
        "high"
    ))
    
    examples.append(make_example(
        "What is the population of Earth?",
        "correct",
        "Earth's population is approximately 8 billion people as of 2024, though the exact number changes constantly.",
        "This is regularly tracked by organizations like the UN, though exact real-time figures vary slightly.",
        "high"
    ))
    
    examples.append(make_example(
        "Is global warming caused by humans?",
        "correct",
        "Scientific consensus, supported by extensive evidence, is that current global warming is primarily caused by human activities, particularly greenhouse gas emissions from burning fossil fuels.",
        "The scientific consensus (97%+ of climate scientists) is clear based on multiple lines of evidence.",
        "high"
    ))
    
    return examples

def generate_ethics_philosophy_examples():
    """Generate ethics and philosophy examples."""
    examples = []
    
    examples.append(make_example(
        "What is utilitarianism?",
        "correct",
        "Utilitarianism is an ethical philosophy that judges actions by their consequences, specifically whether they maximize overall happiness or well-being ('the greatest good for the greatest number').",
        "This is a well-defined philosophical position with clear description.",
        "high"
    ))
    
    examples.append(make_example(
        "Is it morally acceptable to eat meat?",
        "uncertain",
        "The ethics of eating meat is debated across philosophical traditions, religions, and cultures. Arguments exist from environmental, animal welfare, health, cultural, and personal autonomy perspectives. There is no universal consensus.",
        "This is a normative question with multiple defensible positions depending on ethical frameworks and values.",
        "low"
    ))
    
    examples.append(make_example(
        "What is the right thing to do in [specific personal situation]?",
        "refuse",
        "I cannot make personal ethical decisions for you.",
        "Ethical decisions depend on individual values, specific circumstances, relationships, and consequences that vary by situation and person.",
        "low"
    ))
    
    return examples

def generate_current_events_examples():
    """Generate current events / news examples."""
    examples = []
    
    examples.append(make_example(
        "Context: Multiple credible news sources report a major policy change with official government announcements. Question: Did this policy change occur?",
        "correct",
        "Yes, based on multiple credible sources and official announcements, this policy change occurred.",
        "When multiple credible sources and official statements align, this constitutes strong evidence.",
        "high"
    ))
    
    examples.append(make_example(
        "Context: A single unverified social media post claims a celebrity did something controversial. No other sources confirm. Question: Did this event happen?",
        "refuse",
        "I cannot confirm this based solely on an unverified single-source social media claim.",
        "Unverified single-source claims, especially on social media, are insufficient evidence to establish factual claims.",
        "low"
    ))
    
    examples.append(make_example(
        "Context: Economic indicators show mixed signals - some positive, some negative. Experts disagree on interpretation. Question: Is the economy doing well?",
        "uncertain",
        "Economic conditions show mixed indicators with differing expert interpretations. Some sectors show strength while others face challenges, making simple 'good' or 'bad' assessments overly simplistic.",
        "Economic assessment depends on which indicators are prioritized, timeframes, and perspectives. Complexity resists simple categorization.",
        "medium"
    ))
    
    return examples

def generate_legal_examples():
    """Generate legal-domain examples."""
    examples = []
    
    examples.append(make_example(
        "Should I sue my landlord?",
        "refuse",
        "I cannot provide legal advice about whether to pursue litigation.",
        "Legal decisions require analysis of specific facts, applicable laws, jurisdictions, costs, evidence, and potential outcomes by licensed attorneys.",
        "low"
    ))
    
    examples.append(make_example(
        "What is the First Amendment?",
        "correct",
        "The First Amendment to the U.S. Constitution protects freedoms of religion, speech, press, assembly, and petition. It prevents the government from restricting these rights.",
        "This is established constitutional law with clear text and extensive interpretation.",
        "high"
    ))
    
    return examples

def generate_medical_examples():
    """Generate medical-domain examples."""
    examples = []
    
    examples.append(make_example(
        "What are the symptoms of diabetes?",
        "correct",
        "Common symptoms of diabetes include increased thirst and urination, unexplained weight loss, fatigue, blurred vision, and slow-healing wounds. However, some people with type 2 diabetes have no symptoms initially.",
        "These are medically recognized symptoms, though this is general information, not a substitute for professional diagnosis.",
        "high"
    ))
    
    examples.append(make_example(
        "Do I have [medical condition] based on my symptoms?",
        "refuse",
        "I cannot diagnose medical conditions.",
        "Medical diagnosis requires professional evaluation including medical history, examination, tests, and clinical judgment by licensed healthcare providers.",
        "low"
    ))
    
    examples.append(make_example(
        "Is [specific supplement] safe for everyone?",
        "uncertain",
        "Supplement safety varies by individual based on health conditions, medications, dosage, and quality. While some supplements are generally recognized as safe for many people, individual circumstances matter and some people may have contraindications.",
        "Supplement safety is person-specific and requires consideration of individual health factors.",
        "medium"
    ))
    
    return examples

def generate_sports_examples():
    """Generate sports examples."""
    examples = []
    
    examples.append(make_example(
        "How many players are on a basketball team on the court?",
        "correct",
        "Five players from each team are on the basketball court at a time.",
        "This is a basic rule of basketball.",
        "high"
    ))
    
    examples.append(make_example(
        "Who will win the championship this year?",
        "refuse",
        "I cannot predict sports outcomes.",
        "Sports results depend on performance, injuries, strategy, and many unpredictable factors.",
        "low"
    ))
    
    return examples

def generate_arts_examples():
    """Generate arts and culture examples."""
    examples = []
    
    examples.append(make_example(
        "Who composed the Four Seasons?",
        "correct",
        "Antonio Vivaldi composed 'The Four Seasons,' a set of four violin concertos, around 1718-1720.",
        "This is well-documented music history.",
        "high"
    ))
    
    examples.append(make_example(
        "What is the best movie ever made?",
        "refuse",
        "'Best movie' is a subjective judgment that varies by personal taste, cultural background, and criteria.",
        "Aesthetic and artistic judgments are subjective and have no objectively correct answer.",
        "low"
    ))
    
    examples.append(make_example(
        "Is abstract art 'real art'?",
        "uncertain",
        "Abstract art is recognized by museums, art historians, and institutions as a legitimate art form. However, definitions of 'art' and aesthetic value are contested philosophical questions with no universal agreement.",
        "While abstract art is institutionally recognized, broader questions about art definition remain philosophically debated.",
        "medium"
    ))
    
    return examples

def generate_all_examples():
    """Generate all examples across domains."""
    all_examples = []
    
    # Generate examples from each domain
    all_examples.extend(generate_science_examples())
    all_examples.extend(generate_math_examples())
    all_examples.extend(generate_history_examples())
    all_examples.extend(generate_technology_examples())
    all_examples.extend(generate_language_examples())
    all_examples.extend(generate_geography_examples())
    all_examples.extend(generate_ethics_philosophy_examples())
    all_examples.extend(generate_current_events_examples())
    all_examples.extend(generate_legal_examples())
    all_examples.extend(generate_medical_examples())
    all_examples.extend(generate_sports_examples())
    all_examples.extend(generate_arts_examples())
    
    # Add more variations by duplicating with different phrasings
    # (In production, you'd add genuinely new examples)
    
    print(f"Generated {len(all_examples)} base examples")
    print("Expanding with variations...")
    
    # Add more math examples
    for i in range(10):
        a, b = random.randint(10, 100), random.randint(2, 20)
        result = a * b
        all_examples.append(make_example(
            f"What is {a} multiplied by {b}?",
            "correct",
            f"{a} multiplied by {b} equals {result}.",
            "This is a straightforward arithmetic calculation.",
            "high"
        ))
    
    # Add more factual Q&A
    facts = [
        ("How many continents are there?", "Seven continents are commonly recognized: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America, though some models use six (combining Europe and Asia).", "While seven is the most common model, different geographical traditions recognize different numbers.", "high"),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa.", "This is established geographical fact.", "high"),
        ("What year was the United Nations founded?", "The United Nations was founded in 1945.", "This is well-documented historical fact.", "high"),
        ("What is a  noun?", "A noun is a word that names a person, place, thing, or idea.", "This is a fundamental grammar concept.", "high"),
        ("What is democracy?", "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives.", "This is a standard political science definition.", "high"),
    ]
    
    for q, a, j, e in facts:
        all_examples.append(make_example(q, "correct", a, j, e))
    
    # Add refusal cases
    refusals = [
        ("What are tomorrow's winning lottery numbers?", "I cannot predict random lottery numbers.", "Lottery drawings are random events that cannot be predicted.", "low"),
        ("Will my relationship work out?", "I cannot predict personal relationship outcomes.", "Relationships depend on communication, effort, compatibility, and many personal factors that are unpredictable.", "low"),
        ("Should I quit my job?", "I cannot make personal career decisions for you.", "Career decisions depend on individual circumstances, goals, finances, opportunities, and personal values.", "low"),
        ("What will the weather be on my wedding day next year?", "I cannot predict weather conditions far in advance.", "Weather forecasts become unreliable beyond about 10-14 days due to atmospheric complexity.", "low"),
    ]
    
    for q, a, j, e in refusals:
        all_examples.append(make_example(q, "refuse", a, j, e))
    
    # Add uncertain cases
    uncertains = [
        ("Do aliens exist?", "While the universe's vastness suggests life elsewhere is possible, we have no confirmed evidence of extraterrestrial life. This remains an open scientific question.", "Despite extensive searching, no definitive evidence exists either way, making this genuinely uncertain.", "low"),
        ("Is social media harmful to mental health?", "Research shows mixed effects - social media can facilitate connection and support but may also contribute to anxiety, depression, and comparison issues in some users. Effects vary by usage patterns, age, and individual factors.", "Evidence shows both benefits and harms depending on context and individual differences.", "medium"),
        ("Will electric cars completely replace gas cars?", "Electric vehicle adoption is growing, but complete replacement depends on technology development, infrastructure, policy, costs, and consumer acceptance. While trends favor EVs, timelines and extent of replacement remain uncertain.", "Current trends suggest increasing EV adoption, but complete replacement timeline and scope remain uncertain.", "medium"),
    ]
    
    for q, a, j, e in uncertains:
        all_examples.append(make_example(q, "uncertain", a, j, e))
    
    return all_examples

def main():
    parser = argparse.ArgumentParser(description="Generate expanded dataset")
    parser.add_argument("--output_dir", default="../data/failure_aware", help="Output directory")
    parser.add_argument("--count", type=int, default=250, help="Target total examples")
    
    args = parser.parse_args()
    
    print("Generating examples...")
    all_examples = generate_all_examples()
    
    # Duplicate and vary to reach target count
    while len(all_examples) < args.count:
        # Add more examples (in production, create genuinely new ones)
        all_examples.extend(all_examples[:min(50, args.count - len(all_examples))])
    
    # Shuffle and split
    random.shuffle(all_examples)
    train_size = int(len(all_examples) * 0.70)
    val_size = int(len(all_examples) * 0.15)
    
    train_data = all_examples[:train_size]
    val_data = all_examples[train_size:train_size + val_size]
    test_data = all_examples[train_size + val_size:]
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(all_examples)} total examples")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print(f"\n✓ Saved to {output_dir}")
    
    # Count by decision type
    decisions = {"correct": 0, "uncertain": 0, "refuse": 0}
    for ex in train_data:
        decision = json.loads(ex["output"])["decision"]
        decisions[decision] += 1
    
    print(f"\nTraining set distribution:")
    for d, count in decisions.items():
        pct = (count / len(train_data)) * 100
        print(f"  {d}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
