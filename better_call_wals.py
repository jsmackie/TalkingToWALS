from collections import defaultdict
import os
import csv
import pickle

class WALS:

    def __init__(self):
        self.languages = dict()
        self.parameters = dict()
        language_headers, languages = read_file('languages.csv')
        for l in languages:
            kwargs = {language_headers[j]: l[j] for j in range(len(language_headers))}
            self.add_language(Language(kwargs))

        # This finds the general areas of research e.g. "phonology", "writing systems"
        area_headers, areas = read_file('areas.csv')
        area_map = dict()
        for a in areas:
            area_map[a[area_headers.index('ID')]] = a[area_headers.index('Name')]

        # This gets the chapter details, which includes which area the chapter belongs in
        # Chapter names overlap with parameter values below
        chapter_headers, chapters = read_file('chapters.csv')
        chapter_map = dict()
        for ch in chapters:
            id = ch[chapter_headers.index('Area_ID')]
            name = ch[chapter_headers.index('Name')]
            chapter_map[name] = area_map[id]

        # Get parameter values which are like chapter titles or feature names e.g "Rythm Types" or "Locus of case marking"
        parameter_headers, parameters = read_file('parameters.csv')
        for p in parameters:
            kwargs = {parameter_headers[j]: p[j] for j in range(len(parameter_headers))}
            try:
                kwargs['area'] = chapter_map[p[parameter_headers.index('Name')]]
            except KeyError:
                kwargs['area'] = 'Other'
            self.add_parameter(Parameter(kwargs))

        # This creates a codebook which maps from parameter ID to values, e.g
        # codebook['13A']['1'] == 'No tones'
        # codebook['79B']['2'] == 'Imperative'
        code_headers, codes = read_file('codes.csv')
        codebook = defaultdict(dict)
        for c in codes:
            id = c[code_headers.index('Parameter_ID')]
            number = c[code_headers.index('Number')]
            description = c[code_headers.index('Description')]
            codebook[id][number] = description
        self.map_values = codebook

        # Now load the values, which map individual languages to parameter values, e.g
        # value['abn']['26A'] == 'predominantly suffixing'
        # That's the value of the Araban language on parameter 26A "Prefixing vs. Suffixing in Inflectional Morphology"
        value_headers, values = read_file('values.csv')
        for v in values:
            language_id = v[value_headers.index('Language_ID')]
            language = self[language_id]

            parameter_id = v[value_headers.index('Parameter_ID')]
            parameter_value_number = v[value_headers.index('Value')]

            code_result = codebook[parameter_id][parameter_value_number]

            language.set_parameter(parameter_id, code_result)

    def add_language(self, language):
        self.languages[language.id] = language

    def add_parameter(self, parameter):
        self.parameters[parameter.id] = parameter

    def get_language_details(self, query):
        results = list()
        for language in self:
            if query in language.parameters:
                results.append(language)
        return language

    def get_parameter_counts(self, area='all', verbose=True):
        parameter_count = dict()

        area = area.lower()
        for id, parameter in self.parameters.items():
            if area != parameter.area.lower():
                if area == 'all':
                    pass
                else:
                    continue
            count = defaultdict(int)
            for language in self:
                value = language.get_parameter(parameter.id)
                if value:  # only count languages that have an entry in WALS for this
                    count[value] += 1
            parameter_count[parameter.name] = count

        if verbose:
            for parameter in sorted(list(parameter_count.keys())):
                count = parameter_count[parameter]
                sorted_values = reversed(sorted([(k, v) for (k, v) in count.items()], key=lambda x: x[-1]))
                print(parameter)
                j = 0
                for sv in sorted_values:
                    print('\t{} : {}'.format(sv[0], sv[1]))
                    j += 1
                    if j > 5:
                        break
        return parameter_count

    def __iter__(self):
        for language in sorted(self.languages):
            yield self.languages[language]

    def __len__(self):
        return len(self.languages)

    def __getitem__(self, id):
        return self.languages[id]

    def save(self, file='wals.pkl'):
        with open(file, mode='wb') as f:
            pickle.dump(self, f)

class Language:

    def __init__(self, kwargs):
        for key,value in kwargs.items():
            setattr(self, key.lower(), value)
        self.parameters = dict(defaultdict(dict))

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):

        try:
            value = self.parameters[name]
        except KeyError:
            if name.isnumeric():
                #sometimes easier to look for '1' instead of '1A' because there is no '1B'
                name = name + 'A'
                try:
                    value = self.parameters[name]
                except KeyError:
                    value = None
            else:
                value = None

        return value

    def __repr__(self):
        return self.name

class Parameter:

    def __init__(self, kwargs):
        for key,value in kwargs.items():
            setattr(self, key.lower(), value)
        self.value = None

    def set(self, new_value):
        self.value = new_value

    def __repr__(self):
        return self.name

def read_file(filename):
    with open(os.path.join(os.getcwd(), 'data', filename), encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        data = [line.strip() for line in f]
        data = csv.reader(data, quotechar='"', delimiter=',')
    return header, data

def init_wals(return_codebook=False):
    wals = WALS()

    #Get basic language data, including name, ISO, area, family, etc.
    language_headers, languages = read_file('languages.csv')
    for l in languages:
        kwargs = {language_headers[j].lower(): l[j] for j in range(len(language_headers))}
        wals.add_language(Language(kwargs))

    # This finds the general areas of research e.g. "phonology", "writing systems"
    area_headers, areas = read_file('areas.csv')
    area_map = dict()
    for a in areas:
        area_map[a[area_headers.index('ID')]] = a[area_headers.index('Name')]

    #This gets the chapter details, which includes which area the chapter belongs in
    #Chapter names overlap with parameter values below
    chapter_headers, chapters = read_file('chapters.csv')
    chapter_map = dict()
    for ch in chapters:
        id = ch[chapter_headers.index('Area_ID')]
        name = ch[chapter_headers.index('Name')]
        chapter_map[name] = area_map[id]

    #Get parameter values which are like chapter titles or feature names e.g "Rythm Types" or "Locus of case marking"
    parameter_headers, parameters = read_file('parameters.csv')
    for p in parameters:
        kwargs = {parameter_headers[j]: p[j] for j in range(len(parameter_headers))}
        try:
            kwargs['area'] = chapter_map[p[parameter_headers.index('Name')]]
        except KeyError:
            kwargs['area'] = 'Other'
        wals.add_parameter(Parameter(kwargs))

    #This creates a codebook which maps from parameter ID to values, e.g
    #codebook['13A']['1'] == 'No tones'
    #codebook['79B']['2'] == 'Imperative'
    code_headers, codes = read_file('codes.csv')
    codebook = defaultdict(dict)
    for c in codes:
        id = c[code_headers.index('Parameter_ID')]
        number = c[code_headers.index('Number')]
        description = c[code_headers.index('Description')]
        codebook[id][number] = description

    #Now load the values, which map individual languages to parameter values, e.g
    #value['abn']['26A'] == 'predominantly suffixing'
    #That's the value of the Araban language on parameter 26A "Prefixing vs. Suffixing in Inflectional Morphology"
    value_headers, values = read_file('values.csv')
    for v in values:
        language_id = v[value_headers.index('Language_ID')]
        language = wals[language_id]

        parameter_id = v[value_headers.index('Parameter_ID')]
        parameter_value_number = v[value_headers.index('Value')]

        code_result = codebook[parameter_id][parameter_value_number]

        language.set_parameter(parameter_id, code_result)

    if return_codebook:
        return codebook
    else:
        return wals

def extract_typology_data():
    wals = init_wals()

    with open('data/typology_data.txt', mode='w', encoding='utf-8') as f:
        for l_name,language in wals.languages.items():
            print(f'Here are a few (more) typological values for {language.name} found in WALS:', file=f)
            count = 0
            for id, p_value in language.parameters.items():
                count += 1
                if count > 5:
                    print('\n*-----*\n', file=f)
                    print(f'Here are a few (more) typological features of {language.name} found in WALS:', file=f)
                    count = 0
                p_name = wals.parameters[id].name
                print(f'{p_name.capitalize()}, chapter {id}, is {p_value.lower()}', file=f)
            print('\n*-----*\n', file=f)

def map_values():
    wals = WALS()
    descriptions = list()

    for id_, map_ in wals.map_values.items():
        title = wals.parameters[id_].name
        num_values = len(map_)
        list_of_values = ', '.join([f'{number}. '+map_[str(number)] for number in range(1, num_values+1)])
        descriptions.append(f"Map {id_} is titled {title}, and it has the following {num_values} values: {list_of_values}")

    descriptions = '\n-----\n'.join(descriptions)
    with open('./data/map_values.txt', encoding='utf-8', mode='w') as f:
        f.write(descriptions)



if __name__ == '__main__':
    #codebook = init_wals(return_codebook=True)
    from collections import defaultdict
    w = WALS()
    c = w.get_parameter_counts(area='all', verbose=False)#area='Phonology', verbose=True)
    print(c)


