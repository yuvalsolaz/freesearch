import os
import openai



openai.api_key= os.environ.get('OPENAI_API_KEY')
print(openai.api_key)

openai.base_url = "https://..."
openai.default_headers = {"x-foo": "true"}
#
# __messages=[
#         "Below is a table with information about hotel rooms. "
#         "Return a JSON list with an entry for each column. Each entry should have "
#         '{"name": "column name", "description": "column description", "type": "column data type"}'
#         f"\n\n{latest_price.head()}\n\nJSON:\n"
# ]

messages=[
        {
            "role": "system",
            "content": "How do I output all files in a directory using Python?",
        },
    ]

completion = openai.chat.completions.create(
    model="gpt-4",
    response_format={ "type": "json_object" },
    messages=messages
)

print(completion.choices[0].message.content)
