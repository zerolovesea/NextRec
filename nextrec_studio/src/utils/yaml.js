import yaml from 'js-yaml';

export function dumpYaml(obj) {
  return yaml.dump(obj, {
    lineWidth: 120,
    noRefs: true,
    sortKeys: false
  });
}

export function parseYaml(text) {
  if (!text || !text.trim()) {
    return null;
  }
  return yaml.load(text);
}
