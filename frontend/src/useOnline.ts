import { useEffect, useState } from 'react'

/**
 * Whether the browser thinks it has a connection.
 *
 * navigator.onLine is a weak signal — it means "there is a network interface",
 * not "the server is reachable", so it happily reports true on a wifi network
 * with no route out. It is right often enough to be worth showing, and every
 * place that uses it here is paired with a real request that will fail properly
 * if it turns out to be lying. It is never used to claim things are working.
 */
export function useOnline(): boolean {
  const [online, setOnline] = useState(
    () => (typeof navigator === 'undefined' ? true : navigator.onLine),
  )

  useEffect(() => {
    const up = () => setOnline(true)
    const down = () => setOnline(false)
    window.addEventListener('online', up)
    window.addEventListener('offline', down)
    return () => {
      window.removeEventListener('online', up)
      window.removeEventListener('offline', down)
    }
  }, [])

  return online
}
