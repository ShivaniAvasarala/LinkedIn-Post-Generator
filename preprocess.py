import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def process_posts(raw_file_path, processed_file_path=None):
    enriched_posts = []
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)


def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Spanish
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       1. Tags are unified and merged to create a shorter list. 
       Example 1: ["AI", "Artificial Intelligence"] → ["Artificial Intelligence"]
       Example 2: ["ML", "Machine Learning", "Deep Learning"] → ["Machine Learning", "Deep Learning"]
       Example 3: ["Big Data", "Large Scale Data", "Massive Data"] → ["Big Data"]
       Example 4: ["Data Engineering", "Data Pipelines", "ETL"] → ["Data Engineering"]
       Example 5: ["Data Visualization", "Charts and Graphs", "Data Storytelling"] → ["Data Visualization"]
       Example 6: ["Natural Language Processing", "NLP", "Text Processing"] → ["Natural Language Processing"]
    2. 2. Synonyms should be merged into the most common or official term.
       Example: ["NLP", "Natural Language Processing"] → ["Natural Language Processing"]   
    3. Each tag should be follow title case convention. example: "Motivation", "Feature Engineering"
    4. Output should be a JSON object, No preamble
    5. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    6. Acronyms should be expanded where necessary.
        Example 1: ["CV", "Computer Vision"] → ["Computer Vision"]
        Example 2: ["DL", "Deep Learning"] → ["Deep Learning"]   
    7. Tags should be written in their widely accepted form.
        Example 1: ["lstm networks", "long short term memory"] → ["Long Short-Term Memory (LSTM)"]
        Example 2: ["BERT model", "Bidirectional Encoder Representations from Transformers"] → ["BERT"]    
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")