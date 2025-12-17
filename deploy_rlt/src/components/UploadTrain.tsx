import { useEffect, useMemo, useState, FormEvent, ChangeEvent } from 'react'
import axios, { AxiosError } from 'axios'

interface ModelBenchmark {
  model: string
  [key: string]: any
}

interface DSO1Result {
  best_model: string
  metrics_full_data: Record<string, number>
  all_models_benchmark: ModelBenchmark[]
  n_models_tested: number
  top_model: Record<string, any> | null
  all_models: string[]
}

interface MutingInfo {
  enabled: boolean
  available: boolean
  models: string[]
  results_url: string
  plot_url_example: string
  best_model_data?: {
    model_name: string
    columns: string[]
    data: Array<Record<string, any>>
  }
}

interface DSO2Result {
  best_model: string
  metrics_best_model: Record<string, number>
  all_models_benchmark: ModelBenchmark[]
  n_models_tested: number
  top_model: Record<string, any> | null
  all_models: string[]
  muting?: MutingInfo
}

interface ComparisonResult {
  metric_used: string
  dso1_best_score: number
  dso2_best_score: number
  improvement_percent: number
  winner: string
}

interface TrainResult {
  status: string
  message: string
  filename: string
  problem_type: string
  problem_type_detection: string
  target_col: string
  n_rows: number
  n_features: number
  dso1: DSO1Result
  dso2: DSO2Result
  comparison: ComparisonResult
  model_path: string
  eda_dashboard_url: string
  benchmark_results_url: string
}

interface ErrorResponse {
  detail: string
}

interface Props {
  onTrainSuccess: () => void
}

type MutingCurveResponse = {
  status: string
  model_name: string
  columns: string[]
  data: Array<Record<string, any>>
}

export default function UploadTrain({ onTrainSuccess }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [problemType, setProblemType] = useState<string>('')

  // NEW: DSO2 muting params
  const [progressiveMuting, setProgressiveMuting] = useState<boolean>(true)
  const [mutingMinFeatures, setMutingMinFeatures] = useState<number>(5)

  const [loading, setLoading] = useState<boolean>(false)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [error, setError] = useState<string>('')
  const [activeTab, setActiveTab] = useState<'dso1' | 'dso2' | 'comparison'>('dso1')

  // NEW: muting UI state
  const [selectedMutingModel, setSelectedMutingModel] = useState<string>('')
  const [selectedMutingMetric, setSelectedMutingMetric] = useState<string>('')
  const [showMutingTable, setShowMutingTable] = useState<boolean>(false)
  const [mutingTableLoading, setMutingTableLoading] = useState<boolean>(false)
  const [mutingTableError, setMutingTableError] = useState<string>('')
  const [mutingTable, setMutingTable] = useState<MutingCurveResponse | null>(null)

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setError('')
    }
  }

  const mutingMetricOptions = useMemo(() => {
    if (!result) return []
    if (result.problem_type === 'classification') {
      return ['Accuracy', 'F1', 'Precision', 'Recall']
    }
    return ['R2', 'RMSE', 'MAE']
  }, [result])

  // When training result arrives, init muting selects
  useEffect(() => {
    if (!result) return

    const muting = result.dso2.muting
    if (muting?.available && muting.models && muting.models.length > 0) {
      const best = result.dso2.best_model
      const defaultModel = muting.models.includes(best) ? best : muting.models[0]
      setSelectedMutingModel(defaultModel)
    } else {
      setSelectedMutingModel('')
    }

    // default metric depends on task
    const defaultMetric = result.problem_type === 'classification' ? 'Accuracy' : 'R2'
    setSelectedMutingMetric(defaultMetric)

    // reset table
    setShowMutingTable(false)
    setMutingTable(null)
    setMutingTableError('')
  }, [result])

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!file) {
      setError('Veuillez sélectionner un fichier CSV')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    const params: Record<string, any> = {}
    if (problemType) params.problem_type = problemType

    // NEW: send muting params
    params.progressive_muting = progressiveMuting
    params.muting_min_features = mutingMinFeatures

    try {
      const response = await axios.post<TrainResult>(
        '/upload-and-train',
        formData,
        { params }
      )

      console.log('Résultat reçu:', response.data)
      setResult(response.data)
      onTrainSuccess()
    } catch (err) {
      const axiosError = err as AxiosError<ErrorResponse>
      setError(
        axiosError.response?.data?.detail ||
        'Erreur lors de l\'entraînement du modèle'
      )
    } finally {
      setLoading(false)
    }
  }

  const renderModelTable = (models: ModelBenchmark[], title: string, bestModel: string) => {
    if (!models || models.length === 0) {
      return <p className="no-data">⚠️ Aucune donnée disponible</p>
    }

    return (
      <>
        <h4>{title} ({models.length} modèles)</h4>
        <div className="table-container">
          <table className="models-table">
            <thead>
              <tr>
                <th className="rank-column">#</th>
                {Object.keys(models[0]).map((key) => (
                  <th key={key}>
                    {key === 'model' ? 'Modèle' :
                      key.startsWith('test_') ? key.replace('test_', 'Test ').replace('_', ' ') :
                      key.startsWith('cv_') ? key.replace('cv_', 'CV ').replace('_', ' ') :
                      key.replace('_', ' ')
                    }
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((row, idx) => (
                <tr
                  key={idx}
                  className={row.model === bestModel ? 'best-model-row' : ''}
                >
                  <td className="rank-column">
                    {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : idx + 1}
                  </td>
                  {Object.entries(row).map(([key, val], i) => (
                    <td key={i} className={key === 'model' ? 'model-name-cell' : ''}>
                      {typeof val === 'number' ? val.toFixed(4) : String(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  const mutingPlotUrl = useMemo(() => {
    if (!result) return ''
    if (!result.dso2?.muting?.available) return ''
    if (!selectedMutingModel) return ''

    const modelName = encodeURIComponent(selectedMutingModel)
    const metric = selectedMutingMetric ? encodeURIComponent(selectedMutingMetric) : ''

    return `/dso2/muting-plot.png?model_name=${modelName}&metric=${metric}`
  }, [result, selectedMutingModel, selectedMutingMetric])

  const loadMutingTable = async () => {
    if (!result?.dso2?.muting?.available) return
    if (!selectedMutingModel) return

    setMutingTableLoading(true)
    setMutingTableError('')
    setMutingTable(null)

    try {
      const resp = await axios.get<MutingCurveResponse>(
        '/dso2/muting-results',
        { params: { model_name: selectedMutingModel } }
      )
      setMutingTable(resp.data)
    } catch (err) {
      const axiosError = err as AxiosError<ErrorResponse>
      setMutingTableError(
        axiosError.response?.data?.detail || 'Erreur lors du chargement du muting'
      )
    } finally {
      setMutingTableLoading(false)
    }
  }

  const renderMutingTable = () => {
    if (!mutingTable) return null
    if (!mutingTable.data || mutingTable.data.length === 0) {
      return <p className="no-data">⚠️ Courbe vide</p>
    }

    const columns = mutingTable.columns?.length
      ? mutingTable.columns
      : Object.keys(mutingTable.data[0])

    return (
      <div className="table-container" style={{ marginTop: 12 }}>
        <table className="models-table">
          <thead>
            <tr>
              {columns.map((c) => <th key={c}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {mutingTable.data.map((row, idx) => (
              <tr key={idx}>
                {columns.map((c) => {
                  const v = row[c]
                  return (
                    <td key={c}>
                      {typeof v === 'number' ? v.toFixed(4) : String(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }



  return (
    <div className="upload-train">
      <h2>🚀 Upload & Train</h2>
      <p>Téléchargez votre dataset pour lancer l'entraînement</p>

      {/* FORMULAIRE */}
      <div className="train-container">
        <div className="train-form-column">
          <form onSubmit={handleSubmit}>
            {/* DATASET */}
            <div className="form-group">
              <label htmlFor="file-upload">
                📂 Dataset
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                required
              />
              <small>Sélectionnez votre fichier d'entraînement</small>
            </div>

            {/* PROBLEM TYPE */}
            <div className="form-group">
              <label htmlFor="problem-type">
                🔧 Problem type
              </label>
              <select
                id="problem-type"
                value={problemType}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => setProblemType(e.target.value)}
              >
                <option value="">Auto (détection automatique)</option>
                <option value="regression">Régression</option>
                <option value="classification">Classification</option>
              </select>
              <small>Le système détecte automatiquement le type de problème si non spécifié</small>
            </div>

           
            {/* SUBMIT BUTTON */}
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? (
                <>
                  <span className="loading-spinner"></span>
                  Entraînement en cours
                </>
              ) : (
                <>
                  🚀 Lancer l'entraînement complet
                </>
              )}
            </button>
          </form>

          {/* ERROR */}
          {error && (
            <div className="error">
              ❌ {error}
            </div>
          )}
        </div>
      </div>

      {/* RESULTS */}
      {result && (
        <div className="card">
          <h3>✅ {result.message}</h3>

          {/* METRICS GRID */}
          <div className="metrics">
            <div className="metric-card">
              <div className="metric-label">📁 Dataset</div>
              <div className="metric-value">{result.filename}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">📊 Lignes</div>
              <div className="metric-value">{result.n_rows.toLocaleString()}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">🔢 Features</div>
              <div className="metric-value">{result.n_features}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">🎯 Cible</div>
              <div className="metric-value">{result.target_col}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">🔧 Problème</div>
              <div className="metric-value">
                {result.problem_type}
                <span className="badge">
                  {result.problem_type_detection === 'auto' ? '🤖 Auto' : '👤 Manuel'}
                </span>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">🏆 Gagnant</div>
              <div className="metric-value">
                <span className={`winner-badge ${result.comparison.winner.toLowerCase()}`}>
                  {result.comparison.winner}
                </span>
              </div>
            </div>
          </div>

          {/* TABS NAVIGATION */}
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'dso1' ? 'active' : ''}`}
              onClick={() => setActiveTab('dso1')}
            >
              🔵 DSO1 - Standard
            </button>
            <button
              className={`tab ${activeTab === 'dso2' ? 'active' : ''}`}
              onClick={() => setActiveTab('dso2')}
            >
              🟢 DSO2 - RLT Optimisé
            </button>
            <button
              className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
              onClick={() => setActiveTab('comparison')}
            >
              📊 Comparaison
            </button>
          </div>

          {/* TAB CONTENT: DSO1 */}
          {activeTab === 'dso1' && (
            <div className="tab-content">
              <div className="dso-header">
                <h3>🔵 DSO1 - Benchmark Standard</h3>
                <p className="dso-description">
                  Approche classique avec modèles de base et sélection de features RLT
                </p>
              </div>

              <div className="best-model-banner dso1">
                <span className="banner-icon">🏆</span>
                <div className="banner-content">
                  <div className="banner-title">Meilleur modèle DSO1</div>
                  <div className="banner-value">{result.dso1.best_model}</div>
                </div>
              </div>

              {renderModelTable(
                result.dso1.all_models_benchmark,
                '🏆 Tous les modèles DSO1',
                result.dso1.best_model
              )}

              <h4>📈 Métriques du meilleur modèle (données complètes)</h4>
              <div className="metrics-json">
                <pre>{JSON.stringify(result.dso1.metrics_full_data, null, 2)}</pre>
              </div>

              <div className="models-list">
                <h4>📋 Liste des modèles testés</h4>
                <ul>
                  {result.dso1.all_models.map((model, idx) => (
                    <li key={idx}>
                      {model === result.dso1.best_model ? '🏆 ' : ''}
                      {model}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* TAB CONTENT: DSO2 */}
          {activeTab === 'dso2' && (
            <div className="tab-content">
              <div className="dso-header">
                <h3>🟢 DSO2 - Benchmark Optimisé</h3>
                <p className="dso-description">
                  Feature engineering avancé avec combinaisons linéaires pondérées par Variable Importance
                </p>
              </div>

              <div className="best-model-banner dso2">
                <span className="banner-icon">🏆</span>
                <div className="banner-content">
                  <div className="banner-title">Meilleur modèle DSO2</div>
                  <div className="banner-value">{result.dso2.best_model}</div>
                </div>
              </div>

              {renderModelTable(
                result.dso2.all_models_benchmark,
                '🏆 Tous les modèles DSO2',
                result.dso2.best_model
              )}

              <h4>📈 Métriques du meilleur modèle optimisé</h4>
              <div className="metrics-json">
                <pre>{JSON.stringify(result.dso2.metrics_best_model, null, 2)}</pre>
              </div>

              <div className="models-list">
                <h4>📋 Liste des modèles testés</h4>
                <ul>
                  {result.dso2.all_models.map((model, idx) => (
                    <li key={idx}>
                      {model === result.dso2.best_model ? '🏆 ' : ''}
                      {model}
                    </li>
                  ))}
                </ul>
              </div>

              {/* NEW: MUTING SECTION */}
              <div style={{ marginTop: 18 }}>
                <h4>🔇 Muting progressif (DSO2)</h4>

                {!result.dso2.muting?.enabled && (
                  <p className="no-data">ℹ️ Muting désactivé lors de l'entraînement.</p>
                )}

                {result.dso2.muting?.enabled && !result.dso2.muting?.available && (
                  <p className="no-data">⚠️ Muting activé mais aucune courbe n'est disponible.</p>
                )}

                

                {result.dso2.muting?.available && (
                  <>
                    <div className="form-group" style={{ marginTop: 8 }}>
                      <label>Modèle</label>
                      <select
                        value={selectedMutingModel}
                        onChange={(e) => {
                          setSelectedMutingModel(e.target.value)
                          setShowMutingTable(false)
                          setMutingTable(null)
                          setMutingTableError('')
                        }}
                      >
                        {(result.dso2.muting?.models || []).map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Métrique</label>
                      <select
                        value={selectedMutingMetric}
                        onChange={(e) => setSelectedMutingMetric(e.target.value)}
                      >
                        {mutingMetricOptions.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                      <small>Choisis une métrique pour tracer l'évolution pendant le muting.</small>
                    </div>

                    {mutingPlotUrl && (
                      <div className="card" style={{ marginTop: 10 }}>
                        <h4>📉 Courbe muting</h4>
                        <img
                          src={mutingPlotUrl}
                          alt="Muting progressif"
                          style={{ width: '100%', height: 'auto', borderRadius: 10, border: '1px solid #eee' }}
                        />
                        <small style={{ display: 'block', marginTop: 8, color: '#666' }}>
                          Source: {mutingPlotUrl}
                        </small>
                      </div>
                    )}

                    <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={async () => {
                          const next = !showMutingTable
                          setShowMutingTable(next)
                          if (next) await loadMutingTable()
                        }}
                        disabled={!selectedMutingModel}
                      >
                        {showMutingTable ? 'Masquer la table' : 'Afficher la table (JSON)'}
                      </button>

                      {mutingTableLoading && <span>Chargement...</span>}
                      {mutingTableError && <span className="error">❌ {mutingTableError}</span>}
                    </div>

                    {showMutingTable && renderMutingTable()}
                  </>
                )}
              </div>
            </div>
          )}

          {/* TAB CONTENT: COMPARISON */}
          {activeTab === 'comparison' && (
            <div className="tab-content">
              <h3>📊 Comparaison DSO1 vs DSO2</h3>

              <div className="comparison-grid">
                <div className="comparison-card dso1">
                  <div className="card-header">
                    <h4>🔵 DSO1 - Standard</h4>
                  </div>
                  <div className="card-body">
                    <div className="metric-display">
                      <div className="metric-name">{result.comparison.metric_used}</div>
                      <div className="metric-score">{result.comparison.dso1_best_score.toFixed(4)}</div>
                    </div>
                    <div className="model-info">
                      <strong>Modèle:</strong> {result.dso1.best_model}
                    </div>
                    <div className="models-count">
                      <strong>Modèles testés:</strong> {result.dso1.n_models_tested}
                    </div>
                  </div>
                </div>

                <div className="comparison-vs">
                  <div className="vs-circle">VS</div>
                  <div className="improvement-indicator">
                    {result.comparison.improvement_percent > 0 ? (
                      <div className="improvement positive">
                        ⬆️ +{result.comparison.improvement_percent.toFixed(2)}%
                      </div>
                    ) : (
                      <div className="improvement negative">
                        ⬇️ {result.comparison.improvement_percent.toFixed(2)}%
                      </div>
                    )}
                  </div>
                </div>

                <div className="comparison-card dso2">
                  <div className="card-header">
                    <h4>🟢 DSO2 - Optimisé</h4>
                  </div>
                  <div className="card-body">
                    <div className="metric-display">
                      <div className="metric-name">{result.comparison.metric_used}</div>
                      <div className="metric-score">{result.comparison.dso2_best_score.toFixed(4)}</div>
                    </div>
                    <div className="model-info">
                      <strong>Modèle:</strong> {result.dso2.best_model}
                    </div>
                    <div className="models-count">
                      <strong>Modèles testés:</strong> {result.dso2.n_models_tested}
                    </div>
                  </div>
                </div>
              </div>

              <div className={`winner-announcement ${result.comparison.winner.toLowerCase()}`}>
                <h3>
                  {result.comparison.winner === 'DSO2' ? '🎉' : '📊'}
                  {' '}Gagnant: {result.comparison.winner}
                  {' '}
                  {result.comparison.winner === 'DSO2' ? '🎉' : ''}
                </h3>
                <p>
                  {result.comparison.winner === 'DSO2'
                    ? `L'approche optimisée DSO2 améliore les performances de ${Math.abs(result.comparison.improvement_percent).toFixed(2)}%`
                    : `L'approche standard DSO1 reste la meilleure avec ${Math.abs(result.comparison.improvement_percent).toFixed(2)}% d'avantage`
                  }
                </p>
              </div>

              <div className="comparison-details">
                <h4>🔍 Détails de la comparaison</h4>
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Critère</th>
                      <th>DSO1</th>
                      <th>DSO2</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Meilleur modèle</td>
                      <td>{result.dso1.best_model}</td>
                      <td>{result.dso2.best_model}</td>
                    </tr>
                    <tr>
                      <td>{result.comparison.metric_used}</td>
                      <td>{result.comparison.dso1_best_score.toFixed(4)}</td>
                      <td>{result.comparison.dso2_best_score.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <td>Modèles testés</td>
                      <td>{result.dso1.n_models_tested}</td>
                      <td>{result.dso2.n_models_tested}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
