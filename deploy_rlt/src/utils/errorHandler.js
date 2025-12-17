export const getErrorMessage = (err) => {
  if (err.response?.data?.detail) {
    return err.response.data.detail
  }
  if (err.message) {
    return err.message
  }
  return 'Une erreur est survenue'
}
