import { render, screen } from '@testing-library/react';
import Review from './review';

test('renders learn react link', () => {
  render(<Review />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
