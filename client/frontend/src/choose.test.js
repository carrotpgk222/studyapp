import { render, screen } from '@testing-library/react';
import Choose from './choose';

test('renders learn react link', () => {
  render(<Choose />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
