import React, { useState } from 'react';

export interface Parameter {
  name: string;
  type: 'integer' | 'float' | 'categorical';
  min?: number;
  max?: number;
  default?: number;
  choices?: string[];
}

interface ConfigSpaceFormProps {
  onNext: (configSpace: Record<string, any>) => void;
  initialData?: Record<string, any>;
}

const ConfigSpaceForm: React.FC<ConfigSpaceFormProps> = ({ onNext, initialData }) => {
  const [inputMode, setInputMode] = useState<'manual' | 'upload'>('manual');
  const [parameters, setParameters] = useState<Parameter[]>(
    initialData ? Object.entries(initialData).map(([name, def]: [string, any]) => ({
      name,
      type: def.type,
      min: def.min,
      max: def.max,
      default: def.default,
      choices: def.choices,
    })) : []
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedConfig, setUploadedConfig] = useState<Record<string, any> | null>(null);

  const addParameter = () => {
    setParameters([
      ...parameters,
      { name: '', type: 'integer', min: 0, max: 100, default: 50 },
    ]);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const updateParameter = (index: number, field: keyof Parameter, value: any) => {
    const newParams = [...parameters];
    newParams[index] = { ...newParams[index], [field]: value };
    setParameters(newParams);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.json')) {
      alert('Please upload a JSON file');
      return;
    }

    try {
      const text = await file.text();
      const config = JSON.parse(text);

      // Validate the config format
      if (typeof config !== 'object' || Array.isArray(config)) {
        alert('Invalid config space format. Expected a JSON object.');
        return;
      }

      setUploadedFile(file);
      setUploadedConfig(config);

      // Convert uploaded config to parameters for preview
      const params: Parameter[] = Object.entries(config).map(([name, def]: [string, any]) => ({
        name,
        type: def.type,
        min: def.min,
        max: def.max,
        default: def.default,
        choices: def.choices,
      }));
      setParameters(params);
    } catch (error) {
      alert('Failed to parse JSON file: ' + (error instanceof Error ? error.message : 'Unknown error'));
      setUploadedFile(null);
      setUploadedConfig(null);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (inputMode === 'upload') {
      // Use uploaded config
      if (!uploadedConfig) {
        alert('Please upload a config space JSON file');
        return;
      }
      onNext(uploadedConfig);
      return;
    }

    // Manual mode validation
    if (parameters.length === 0) {
      alert('Please add at least one parameter');
      return;
    }

    for (const param of parameters) {
      if (!param.name.trim()) {
        alert('All parameters must have a name');
        return;
      }
      if (param.type === 'categorical' && (!param.choices || param.choices.length === 0)) {
        alert(`Parameter "${param.name}" must have choices`);
        return;
      }
      if ((param.type === 'integer' || param.type === 'float') &&
        (param.min === undefined || param.max === undefined)) {
        alert(`Parameter "${param.name}" must have min and max values`);
        return;
      }
    }

    // Convert to config space format
    const configSpace: Record<string, any> = {};
    parameters.forEach((param) => {
      const config: any = { type: param.type };
      if (param.type === 'integer' || param.type === 'float') {
        config.min = param.min;
        config.max = param.max;
        if (param.default !== undefined) {
          config.default = param.default;
        }
      } else if (param.type === 'categorical') {
        config.choices = param.choices;
        if (param.default !== undefined) {
          config.default = param.default;
        }
      }
      configSpace[param.name] = config;
    });

    onNext(configSpace);
  };

  return (
    <form onSubmit={handleSubmit} className="config-space-form">
      <div className="form-header">
        <h3>Step 1: Configure Parameter Space</h3>
        <p className="form-description">Define the parameters for your configuration space</p>
      </div>

      {/* Input Mode Selector */}
      <div className="input-mode-selector">
        <label className={`mode-option ${inputMode === 'manual' ? 'active' : ''}`}>
          <input
            type="radio"
            name="inputMode"
            value="manual"
            checked={inputMode === 'manual'}
            onChange={(e) => setInputMode(e.target.value as 'manual' | 'upload')}
          />
          <span className="mode-label">
            <span className="mode-icon">✏️</span>
            Manual Configuration
          </span>
        </label>
        <label className={`mode-option ${inputMode === 'upload' ? 'active' : ''}`}>
          <input
            type="radio"
            name="inputMode"
            value="upload"
            checked={inputMode === 'upload'}
            onChange={(e) => setInputMode(e.target.value as 'manual' | 'upload')}
          />
          <span className="mode-label">
            <span className="mode-icon">📁</span>
            Upload JSON File
          </span>
        </label>
      </div>

      {/* Upload Mode */}
      {inputMode === 'upload' && (
        <div className="upload-mode">
          <div className="upload-zone">
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              className="file-input-hidden"
              id="config-file-input"
            />
            <label htmlFor="config-file-input" className="upload-label">
              <div className="upload-icon">📄</div>
              <p className="upload-text">
                {uploadedFile ? uploadedFile.name : 'Click to select config_space.json'}
              </p>
              {uploadedFile && (
                <p className="file-size">
                  ({(uploadedFile.size / 1024).toFixed(2)} KB)
                </p>
              )}
            </label>
          </div>

          {uploadedConfig && (
            <div className="config-preview">
              <h4>Preview ({Object.keys(uploadedConfig).length} parameters)</h4>
              <div className="preview-list">
                {Object.entries(uploadedConfig).map(([name, def]: [string, any]) => (
                  <div key={name} className="preview-item">
                    <span className="param-name">{name}</span>
                    <span className="param-type">{def.type}</span>
                    {def.min !== undefined && def.max !== undefined && (
                      <span className="param-range">
                        [{def.min}, {def.max}]
                      </span>
                    )}
                    {def.choices && (
                      <span className="param-choices">
                        {def.choices.join(', ')}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="info-box">
            <h4>JSON Format Example</h4>
            <pre>{`{
  "spark.executor.cores": {
    "type": "integer",
    "min": 1,
    "max": 8
  },
  "spark.memory.fraction": {
    "type": "float",
    "min": 0.1,
    "max": 0.9
  },
  "spark.shuffle.compress": {
    "type": "categorical",
    "choices": ["true", "false"]
  }
}`}</pre>
          </div>
        </div>
      )}

      {/* Manual Mode */}
      {inputMode === 'manual' && (
        <>
          <div className="parameters-list">
            {parameters.map((param, index) => (
              <div key={index} className="parameter-item">
                <div className="parameter-row">
                  <div className="form-field">
                    <label>Parameter Name</label>
                    <input
                      type="text"
                      value={param.name}
                      onChange={(e) => updateParameter(index, 'name', e.target.value)}
                      placeholder="e.g., spark.executor.cores"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label>Type</label>
                    <select
                      value={param.type}
                      onChange={(e) =>
                        updateParameter(index, 'type', e.target.value as Parameter['type'])
                      }
                    >
                      <option value="integer">Integer</option>
                      <option value="float">Float</option>
                      <option value="categorical">Categorical</option>
                    </select>
                  </div>

                  <button
                    type="button"
                    className="btn-remove"
                    onClick={() => removeParameter(index)}
                    title="Remove parameter"
                  >
                    ×
                  </button>
                </div>

                {(param.type === 'integer' || param.type === 'float') && (
                  <div className="parameter-row">
                    <div className="form-field">
                      <label>Min</label>
                      <input
                        type="number"
                        step={param.type === 'float' ? '0.01' : '1'}
                        value={param.min ?? ''}
                        onChange={(e) =>
                          updateParameter(index, 'min', parseFloat(e.target.value))
                        }
                        required
                      />
                    </div>

                    <div className="form-field">
                      <label>Max</label>
                      <input
                        type="number"
                        step={param.type === 'float' ? '0.01' : '1'}
                        value={param.max ?? ''}
                        onChange={(e) =>
                          updateParameter(index, 'max', parseFloat(e.target.value))
                        }
                        required
                      />
                    </div>

                    <div className="form-field">
                      <label>Default (optional)</label>
                      <input
                        type="number"
                        step={param.type === 'float' ? '0.01' : '1'}
                        value={param.default ?? ''}
                        onChange={(e) =>
                          updateParameter(index, 'default', parseFloat(e.target.value))
                        }
                      />
                    </div>
                  </div>
                )}

                {param.type === 'categorical' && (
                  <div className="parameter-row">
                    <div className="form-field full-width">
                      <label>Choices (comma-separated)</label>
                      <input
                        type="text"
                        value={param.choices?.join(', ') ?? ''}
                        onChange={(e) =>
                          updateParameter(
                            index,
                            'choices',
                            e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                          )
                        }
                        placeholder="e.g., true, false"
                        required
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <button type="button" className="btn btn-add" onClick={addParameter}>
            + Add Parameter
          </button>
        </>
      )}

      <div className="form-actions">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={inputMode === 'manual' ? parameters.length === 0 : !uploadedConfig}
        >
          Next: Configure Steps
        </button>
      </div>
    </form>
  );
};

export default ConfigSpaceForm;
