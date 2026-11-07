# ProjectMilei2

Prueba de concepto de análisis de opinión con **32 visualizaciones interactivas** de 977 respuestas: posiciones políticas, identidades declaradas, diferencias sociodemográficas, medios y evaluaciones de dirigentes.

## Ejecutar localmente

```sh
python -m http.server 8000
```

Abrir `http://localhost:8000`. Usar un servidor HTTP: abrir `index.html` como archivo no permite siempre cargar la base JSON mediante `fetch`.

## Publicación en GitHub Pages

En Settings → Pages, seleccionar **GitHub Actions**. El workflow activo `Publish GitHub Pages` publica la última versión calculada, al subir cambios o al ejecutarlo manualmente.

## Actualización desde Google Sheets

**Estado: preparada, pero desactivada.** La revisión automática bloqueó la ejecución de la actualización por un destino de telemetría no verificado. No se ejecuta ni se programa desde GitHub hasta resolver ese bloqueo. La versión activa del sitio conserva la última base calculada.

La propuesta está en `.github/refresh-sheets.yml.disabled`: ejecución cada hora (minuto 17 UTC), al subir cambios a `main` y manualmente desde Actions. Las ejecuciones programadas de GitHub pueden demorarse; no es sincronización instantánea.

Lee las respuestas y la clave de puntajes de la hoja configurada en `sheets_source.json`, calcula las 32 figuras y actualiza el sitio. Una huella de los datos evita recalcular si no cambiaron. La web muestra la fecha de la última actualización con cambios.

La hoja ya permite exportación CSV sin autenticación. No se modificaron sus permisos y no se guardan claves. Si deja de permitir esa lectura, la actualización falla y queda publicada la última versión válida. Si cambia un texto de opción o la estructura de columnas, se detiene para revisar la correspondencia en `analysis/form_mapping.json`.

El cálculo usa Python aislado (`-I -S`) y bloquea conexiones de red antes de importar las librerías de análisis. Solo la fase de descarga puede realizar GET al export de Google y sus redirecciones `googleusercontent.com`. Las filas de origen se procesan en un directorio temporal que se descarta; solo se guardan resultados preparados para publicación.

El comando preparado es `python -I -S scripts/refresh.py`; sus dependencias están en `requirements.txt`. La ejecución no fue validada como exitosa y sigue pendiente de autorización por el bloqueo indicado.

## Base de datos pública

- [`data/charts.json`](data/charts.json): figuras y datos calculados que carga la aplicación mediante HTTP.
- [`data/manifest.json`](data/manifest.json): esquema, procedencia, alcance y lista de tablas.
- CSV en `data/`: asociaciones, dispersión, diversidad de preguntas, cruces agregados y clave de puntuación.

Los datos permiten descargar y reutilizar los gráficos online. La publicación no incluye una tabla individual que vincule datos sociodemográficos con respuestas políticas, ni fechas, contactos o nombres de participantes. Las figuras espaciales incluyen posiciones sin identificadores personales. La base original se conserva fuera del repositorio público.

## Lectura

Cada gráfico contiene una explicación, denominadores y limitaciones relevantes. Es un análisis exploratorio de esta base, sin ponderación de representatividad nacional. Las asociaciones no establecen causalidad.

La puntuación respeta la tabla por número de pregunta y opción del formulario actual; consultar `data/clave_auditada.csv`. No se corrigen por interpretación los desajustes temáticos de la clave. Los referentes siguen representados por los centros de sus seguidores: no se han simulado respuestas de dirigentes.

## Archivos

- `index.html`: página y navegación.
- `assets/app.js`: carga de datos, filtros y dibujo progresivo de figuras.
- `assets/plotly.min.js`: Plotly incluido localmente; conserva su aviso de licencia.
- `data/`: base JSON y tablas agregadas.
- `scripts/verify.mjs`: comprobación de archivos y estructura de la publicación.

No necesita backend, claves de API ni conexión con Google Drive para funcionar.
