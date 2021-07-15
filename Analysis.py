#!/usr/bin/env python
import os
import sys
import json
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def verdict_vs_length_of_comments(data, verdict_data):
    for year in data:
        plot_data = {
            'length_of_comments': [],
            'recommendation': [],
            'verdict': [],
        }
        for review in data[year]['reviews']:
            plot_data['length_of_comments'].append(review['length_of_comment'])
            plot_data['verdict'].append(verdict_data[year][review['filename']])
            plot_data['recommendation'].append(review['recommendation'])
            print("LEAVE: ", review['length_of_comment'], verdict_data[year]
                  [review['filename']], review['recommendation'])
        ax = sns.scatterplot(data=plot_data, x='length_of_comments',
                             y='recommendation', hue='verdict')
        plt.tight_layout()
        plt.savefig(f'{year}_verdict_vs_length_of_comments.png')


def generate_wordcloud(text, title):
    print("Generating WordCloud", title)
    wordcloud = WordCloud(max_font_size=40).generate_from_text(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    wordcloud.to_file("WordCloud_" + title + ".png")


def prepare_wordcloud(data, verdict_data):
    for year in data:
        accepted_papers = ""
        accepted_reviews = ""
        rejected_papers = ""
        rejected_reviews = ""
        for paper in data[year]['papers']:
            if verdict_data[year][paper['filename']]:
                accepted_papers += paper['text']
            else:
                rejected_papers += paper['text']

            for review in data[year]['reviews']:
                if verdict_data[year][review['filename']]:
                    accepted_reviews += review['comment']
                else:
                    rejected_reviews += review['comment']
        generate_wordcloud(accepted_papers, f"Accepted Papers_{year}")
        generate_wordcloud(accepted_reviews, f"Accepted Reviews_{year}")
        generate_wordcloud(rejected_papers, f"Rejected Papers_{year}")
        generate_wordcloud(rejected_reviews, f"Rejected Reviews_{year}")


def main():
    directory = sys.argv[1]
    # years = ['2017', '2018', '2019', '2020']
    years = ['2018', '2019', '2020']
    data = {}
    verdict_data = {}
    for year in years:
        data[year] = {
            'papers': [],
            'reviews': [],
        }
        verdict_data[year] = {}
        files = os.listdir(os.path.join(directory, year))
        for file in files:
            with open(os.path.join(directory, year, file), 'r') as f:
                json_data = json.load(f)
            print(f'Loaded {file}')

            if 'paper' in file:
                plot_data = {
                    'filename': json_data['name'][:-4],
                    'text': "",
                    'references_count': 0
                }
                if json_data['metadata'].get('sections', None):
                    for section in json_data['metadata']['sections']:
                        plot_data['text'] += section['text']

                plot_data['references_count'] = len(
                    json_data['metadata']['references'])
                data[year]['papers'].append(plot_data)

            else:
                verdict_data[year][file[:-5]
                                   ] = 1 if json_data['verdict'].lower() == "accept" else 0
                for review in json_data['reviews']:
                    if review['comments']:
                        data[year]['reviews'].append({
                            'recommendation': review['recommendation'],
                            'length_of_comment': len(word_tokenize(review['comments'])),
                            'comment': review['comments'],
                            'filename': file[:-5],
                            'recommendation': review['recommendation']
                        })
    # verdict_vs_length_of_comments(data, verdict_data)
    prepare_wordcloud(data, verdict_data)


if __name__ == "__main__":
    main()
